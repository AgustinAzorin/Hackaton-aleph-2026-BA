/**
 * Núcleo de inferencia on-device. Todo lo que hay acá corre contra `@qvac/sdk`;
 * no hay una sola llamada a un servicio remoto de inferencia.
 *
 * ## Presupuesto de memoria
 *
 * El pipeline está pensado para un portátil de ~8 GB de RAM, así que nunca
 * puede haber dos modelos grandes cargados a la vez. En vez de cargar y
 * descargar por documento (que pagaría el costo de carga N veces), el trabajo
 * se agrupa en tres fases, cada una con un único modelo vivo:
 *
 *   Fase 1 — OCR_LATIN (~98 MB)   : rasteriza y transcribe todos los documentos
 *   Fase 2 — GTE_LARGE_FP16 (~670 MB) : indexa respaldos y recupera candidatos
 *   Fase 3 — QWEN3_4B_INST_Q4_K_M (~2,5 GB) : extrae JSON y audita
 *
 * Por eso la recuperación RAG ocurre en la fase 2 y no dentro de la auditoría:
 * buscar durante la fase 3 exigiría tener el modelo de embeddings y el LLM
 * cargados simultáneamente, que es justo lo que el presupuesto prohíbe.
 */
import {
  loadModel,
  unloadModel,
  ocr,
  completion,
  ragIngest,
  ragSearch,
  ragCloseWorkspace,
  close,
  OCR_LATIN,
  GTE_LARGE_FP16,
  QWEN3_4B_INST_Q4_K_M,
  type ModelProgressUpdate,
  type OCRTextBlock,
  type RagSearchResult
} from '@qvac/sdk'
import { readdir } from 'node:fs/promises'
import path from 'node:path'
import type { z } from 'zod'
import {
  AUDIT_JSON_SCHEMA,
  AuditResultSchema,
  INVOICE_JSON_SCHEMA,
  InvoiceDataSchema,
  type AuditResult,
  type DiscrepancyReport,
  type InvoiceData,
  type ReconciliationVerdict,
  type StageTiming
} from '../types.js'
import { isSupportedDocument, toImagePages } from './rasterize.js'

// ---------------------------------------------------------------------------
// Progreso
// ---------------------------------------------------------------------------

export interface ProgressEvent {
  stage: 'model' | 'ocr' | 'index' | 'extract' | 'audit' | 'done'
  message: string
  current?: number
  total?: number
}

export type OnProgress = (event: ProgressEvent) => void

const noop: OnProgress = () => {}

/** Marca de tiempo legible, para poder mostrar latencias por etapa en la UI. */
function now(): number {
  return Date.now()
}

// ---------------------------------------------------------------------------
// Ciclo de vida de modelos
// ---------------------------------------------------------------------------

/**
 * Carga un modelo, ejecuta el trabajo y lo descarga siempre — incluso si el
 * trabajo lanza. Es la garantía de que nunca queda un modelo ocupando RAM
 * cuando arranca la fase siguiente.
 *
 * La carga se recibe como thunk en vez de como descriptor: así cada fase
 * conserva el tipo literal de su constante de modelo y `loadModel` puede
 * inferir el `modelType` y estrechar el `modelConfig` por motor.
 */
async function withModel<T>(
  label: string,
  load: (onDownload: (progress: ModelProgressUpdate) => void) => Promise<string>,
  onProgress: OnProgress,
  work: (modelId: string) => Promise<T>
): Promise<T> {
  onProgress({ stage: 'model', message: `Cargando ${label}...` })

  const modelId = await load((progress) => {
    onProgress({
      stage: 'model',
      message: `Descargando ${label}: ${progress.percentage.toFixed(0)}%`,
      current: progress.percentage,
      total: 100
    })
  })

  onProgress({ stage: 'model', message: `${label} listo.` })
  try {
    return await work(modelId)
  } finally {
    await unloadModel({ modelId }).catch(() => {
      // Descargar es best-effort: si falla, no debe tapar el error original
      // ni tumbar el pipeline.
    })
    onProgress({ stage: 'model', message: `${label} descargado.` })
  }
}

// ---------------------------------------------------------------------------
// Fase 1 — OCR
// ---------------------------------------------------------------------------

/**
 * Reconstruye el texto en orden de lectura a partir de los bloques del OCR.
 *
 * El motor devuelve bloques sueltos con su bounding box; sin reordenarlos, las
 * columnas de una tabla de factura llegan al modelo entremezcladas y los montos
 * se despegan de su descripción. Se agrupan por banda vertical (misma línea) y
 * dentro de cada banda se ordenan de izquierda a derecha.
 */
export function blocksToText(blocks: OCRTextBlock[]): string {
  const positioned = blocks.filter((b) => b.bbox !== undefined)

  // Sin bounding boxes no hay nada que reordenar: se respeta el orden de llegada.
  if (positioned.length === 0) {
    return blocks
      .map((b) => b.text.trim())
      .filter(Boolean)
      .join('\n')
  }

  const withBox = positioned.map((b) => {
    const [x1, y1, , y2] = b.bbox as [number, number, number, number]
    return { text: b.text.trim(), x: x1, y: y1, height: Math.abs(y2 - y1) }
  })

  // Tolerancia de línea proporcional a la altura típica del texto: dos bloques
  // pertenecen a la misma fila si sus tapas verticales caen dentro de media
  // altura de carácter.
  const medianHeight =
    [...withBox].sort((a, b) => a.height - b.height)[Math.floor(withBox.length / 2)]?.height ?? 10
  const tolerance = Math.max(medianHeight * 0.6, 4)

  const sorted = [...withBox].sort((a, b) => a.y - b.y || a.x - b.x)
  const lines: (typeof sorted)[] = []

  for (const block of sorted) {
    const line = lines[lines.length - 1]
    const anchor = line?.[0]
    if (line && anchor && Math.abs(block.y - anchor.y) <= tolerance) {
      line.push(block)
    } else {
      lines.push([block])
    }
  }

  return lines
    .map((line) =>
      [...line]
        .sort((a, b) => a.x - b.x)
        .map((b) => b.text)
        .filter(Boolean)
        .join('  ')
    )
    .filter((line) => line.trim().length > 0)
    .join('\n')
}

/**
 * Rasteriza (si hace falta) y transcribe un documento completo a texto plano.
 * Esta es la ruta principal exigida por el track: OCR → texto → LLM de texto,
 * sin depender de que entre en memoria un modelo multimodal.
 */
async function ocrDocument(modelId: string, filePath: string): Promise<string> {
  const pages = await toImagePages(filePath)
  const texts: string[] = []

  for (const page of pages) {
    const { blocks } = ocr({ modelId, image: page, options: { paragraph: false } })
    texts.push(blocksToText(await blocks))
  }

  return texts.join('\n\n').trim()
}

/** Lista los documentos soportados de una carpeta, en orden estable. */
export async function listDocuments(folderPath: string): Promise<string[]> {
  let entries
  try {
    entries = await readdir(folderPath, { withFileTypes: true })
  } catch (error) {
    // Una carpeta mal elegida es el error más probable en una demo: conviene
    // decir cuál es y por qué, en vez de propagar un ENOENT crudo.
    const reason =
      error instanceof Error && 'code' in error && error.code === 'ENOENT'
        ? 'la carpeta no existe'
        : error instanceof Error
          ? error.message
          : String(error)
    throw new Error(`No se pudo leer la carpeta "${folderPath}": ${reason}.`)
  }

  return entries
    .filter((e) => e.isFile() && !e.name.startsWith('.') && isSupportedDocument(e.name))
    .map((e) => path.join(folderPath, e.name))
    .sort()
}

export interface OcrDocumentResult {
  file: string
  text: string
  error: string | null
  ms: number
}

/**
 * Transcribe un lote de documentos con un único modelo OCR cargado.
 *
 * Un documento ilegible no interrumpe el lote: se registra su error y el
 * pipeline sigue, para que un archivo corrupto nunca tumbe la corrida.
 */
export async function ocrDocuments(
  files: string[],
  onProgress: OnProgress = noop
): Promise<OcrDocumentResult[]> {
  if (files.length === 0) return []

  return withModel(
    OCR_LATIN.name,
    (onDownload) =>
      loadModel({
        modelSrc: OCR_LATIN,
        modelConfig: {
          langList: ['en'],
          magRatio: 1.5,
          contrastRetry: true,
          lowConfidenceThreshold: 0.5,
          recognizerBatchSize: 1
        },
        onProgress: onDownload
      }),
    onProgress,
    async (modelId) => {
      const results: OcrDocumentResult[] = []

      for (const [index, file] of files.entries()) {
        const name = path.basename(file)
        onProgress({
          stage: 'ocr',
          message: `OCR ${name}`,
          current: index + 1,
          total: files.length
        })

        const started = now()
        try {
          const text = await ocrDocument(modelId, file)
          if (text.length === 0) {
            results.push({
              file,
              text: '',
              error: 'El OCR no devolvió texto legible.',
              ms: now() - started
            })
          } else {
            results.push({ file, text, error: null, ms: now() - started })
          }
        } catch (error) {
          results.push({
            file,
            text: '',
            error: error instanceof Error ? error.message : String(error),
            ms: now() - started
          })
        }
      }

      return results
    }
  )
}

// ---------------------------------------------------------------------------
// Fase 2 — Indexación RAG y recuperación
// ---------------------------------------------------------------------------

/**
 * Consulta compacta para la búsqueda semántica.
 *
 * Se arma con el encabezado de la factura en vez de con el documento entero:
 * ahí viven el proveedor y la referencia de PO, que son la señal que discrimina
 * un respaldo del otro. El cuerpo de la tabla agrega ruido y diluye el embedding.
 */
export function buildRetrievalQuery(invoiceText: string): string {
  return invoiceText.split('\n').slice(0, 12).join('\n').slice(0, 800)
}

export interface IndexAndRetrieveResult {
  /** Documentos de respaldo efectivamente indexados. */
  indexed: string[]
  /** Candidatos recuperados por factura, con su score de similitud. */
  retrieved: Map<string, RagSearchResult[]>
}

/**
 * Indexa los documentos de respaldo y recupera, para cada factura, los
 * fragmentos más parecidos. Todo ocurre con el modelo de embeddings cargado,
 * que se descarga antes de que el LLM entre en memoria.
 */
export async function indexSupportDocuments(
  supportDocs: OcrDocumentResult[],
  invoices: OcrDocumentResult[],
  workspace: string,
  onProgress: OnProgress = noop,
  topK = 3
): Promise<IndexAndRetrieveResult> {
  const usable = supportDocs.filter((d) => d.error === null && d.text.length > 0)
  const retrieved = new Map<string, RagSearchResult[]>()

  if (usable.length === 0) {
    onProgress({ stage: 'index', message: 'No hay documentos de respaldo legibles para indexar.' })
    return { indexed: [], retrieved }
  }

  return withModel(
    GTE_LARGE_FP16.name,
    (onDownload) => loadModel({ modelSrc: GTE_LARGE_FP16, onProgress: onDownload }),
    onProgress,
    async (modelId) => {
    // El nombre del archivo se antepone al contenido para que el fragmento
    // recuperado le diga al auditor de qué documento salió.
    const documents = usable.map((d) => `[${path.basename(d.file)}]\n${d.text}`)

    onProgress({
      stage: 'index',
      message: `Indexando ${documents.length} documentos de respaldo...`,
      total: documents.length
    })

    await ragIngest({
      modelId,
      documents,
      workspace,
      // Una orden de compra es una unidad atómica de pocos cientos de
      // caracteres. Con trozos chicos, el total puede caer en un fragmento
      // distinto al de los ítems y el auditor nunca llega a verlo junto.
      chunkOpts: { chunkSize: 2048, chunkOverlap: 128, chunkStrategy: 'paragraph' },
      progressInterval: 100,
      onProgress: (stage, current, total) => {
        onProgress({ stage: 'index', message: `RAG ${stage}`, current, total })
      }
    })

    for (const [index, invoice] of invoices.entries()) {
      if (invoice.error !== null || invoice.text.length === 0) {
        retrieved.set(invoice.file, [])
        continue
      }

      onProgress({
        stage: 'index',
        message: `Buscando respaldo de ${path.basename(invoice.file)}`,
        current: index + 1,
        total: invoices.length
      })

      try {
        const hits = await ragSearch({
          modelId,
          query: buildRetrievalQuery(invoice.text),
          topK,
          workspace
        })
        retrieved.set(invoice.file, hits)
      } catch {
        // Una búsqueda fallida no invalida la corrida: la factura queda sin
        // evidencia y el auditor la marcará UNCERTAIN.
        retrieved.set(invoice.file, [])
      }
    }

    return { indexed: usable.map((d) => d.file), retrieved }
  })
}

// ---------------------------------------------------------------------------
// Fase 3 — Extracción y auditoría
// ---------------------------------------------------------------------------

interface Message {
  role: 'system' | 'user' | 'assistant'
  content: string
}

type JsonResult<T> = { ok: true; data: T } | { ok: false; error: string }

/**
 * Pide al modelo un JSON que valide contra un schema, con un reintento.
 *
 * La gramática GBNF derivada del JSON Schema ya restringe la generación, pero
 * eso garantiza la forma, no la coherencia semántica; zod es la segunda barrera.
 * Si la validación falla, se reintenta una vez devolviéndole al modelo el error
 * concreto. Si vuelve a fallar, se informa el error hacia arriba en vez de
 * lanzar: el documento termina marcado para revisión humana, nunca en un crash.
 */
async function completeJson<T>(
  modelId: string,
  history: Message[],
  jsonSchema: Record<string, unknown>,
  schemaName: string,
  validator: z.ZodType<T>
): Promise<JsonResult<T>> {
  let messages = history
  let lastError = 'desconocido'

  for (let attempt = 1; attempt <= 2; attempt++) {
    let raw = ''
    try {
      const run = completion({
        modelId,
        history: messages,
        stream: false,
        responseFormat: {
          type: 'json_schema',
          json_schema: { name: schemaName, schema: jsonSchema }
        }
      })
      raw = (await run.final).contentText.trim()
    } catch (error) {
      lastError = error instanceof Error ? error.message : String(error)
      break // Un fallo de inferencia no se arregla reintentando el mismo prompt.
    }

    let parsed: unknown
    try {
      parsed = JSON.parse(raw)
    } catch {
      lastError = `La salida no es JSON parseable: ${raw.slice(0, 200)}`
      messages = [
        ...history,
        { role: 'assistant', content: raw },
        {
          role: 'user',
          content: `Tu respuesta anterior no era JSON válido. Respondé únicamente con un objeto JSON que cumpla el schema. Error: ${lastError}`
        }
      ]
      continue
    }

    const result = validator.safeParse(parsed)
    if (result.success) return { ok: true, data: result.data }

    lastError = result.error.issues
      .map((issue) => `${issue.path.join('.') || '(raíz)'}: ${issue.message}`)
      .join('; ')
    messages = [
      ...history,
      { role: 'assistant', content: raw },
      {
        role: 'user',
        content: `Tu respuesta anterior no cumple el schema. Corregí exactamente estos problemas y respondé sólo con el JSON corregido: ${lastError}`
      }
    ]
  }

  return { ok: false, error: lastError }
}

const EXTRACTION_SYSTEM = `Sos un extractor de datos de facturas. Recibís el texto crudo de una factura obtenido por OCR y devolvés únicamente un objeto JSON.

Reglas:
- Copiá los valores tal como figuran en el documento. No inventes ni completes datos que no estén.
- Los montos van como número, sin símbolo de moneda ni separadores de miles (por ejemplo 2980.00, no "USD 2,980.00").
- "currency" es el código ISO de tres letras (USD, EUR, ARS).
- "totalAmount" es el total impreso en el documento, aunque no coincida con la suma de los ítems. No lo recalcules.
- "poReference" es la orden de compra referenciada; si el documento no menciona ninguna, usá cadena vacía "".
- El OCR puede traer errores; si un campo es ilegible, usá "" para texto y 0 para números.
/no_think`

const AUDIT_SYSTEM = `Sos un asistente de auditoría de cuentas a pagar. Recibís una factura ya extraída y los fragmentos de texto de los documentos de respaldo recuperados. Tu tarea es TRANSCRIBIR lo que dice el respaldo, no juzgar si coincide.

La comparación la hace después un verificador determinista. Tu único trabajo es leer bien. Completá los campos EN ORDEN:

1. "supportDocumentId": el identificador del respaldo que corresponde a esta factura, por ejemplo "PO-5003.pdf". Si ninguno de los fragmentos recuperados corresponde a esta factura, usá "".
2. "supportVendorName": el proveedor que figura en ese respaldo, copiado literal. Este campo es crítico: si el respaldo es de otro proveedor, es la señal de que la recuperación trajo el documento equivocado. No lo copies de la factura — leelo del respaldo.
3. "supportTotalAmount": el TOTAL impreso en el respaldo, como número. Buscalo en el texto. Si no lo encontrás, usá 0. Nunca lo deduzcas del total de la factura.
4. "supportItems": TODOS los ítems que figuran en el respaldo, uno por uno, con su descripción, cantidad, precio unitario e importe. Transcribí la lista completa aunque sea larga: si omitís un ítem, el verificador va a creer que la factura dejó de facturarlo.
5. "discrepancies": dejalo vacío ([]). Lo completa el verificador.
6. "verdict", 7. "confidence", 8. "summary": tu impresión general. El verificador puede corregirla.

Advertencias:
- No inventes valores. Si algo no está en el texto del respaldo, usá "" o 0.
- Los fragmentos recuperados pueden ser de otra factura o de otro proveedor. Si es así, "supportDocumentId" va vacío.
- "summary" es UNA sola frase.
/no_think`

/** Tolerancia de comparación de montos: por debajo de un centavo es redondeo. */
const AMOUNT_EPSILON = 0.01

/**
 * Score de recuperación por debajo del cual el respaldo no se considera
 * evidencia si además no se pudo confirmar el proveedor. Los cruces correctos
 * en los datos de prueba puntúan ~0,86-0,89; un cruce contra otro proveedor,
 * ~0,69.
 */
const WEAK_RETRIEVAL_SCORE = 0.75

/** Sufijos societarios que no aportan nada al comparar dos razones sociales. */
const CORPORATE_SUFFIXES = new Set([
  'llc',
  'inc',
  'co',
  'corp',
  'ltd',
  'limited',
  'sa',
  'srl',
  'gmbh',
  'bv',
  'plc',
  'company'
])

/** Reduce una razón social a sus palabras distintivas. */
function vendorTokens(name: string): Set<string> {
  return new Set(
    name
      .toLowerCase()
      .replace(/[^a-z0-9áéíóúñü\s]/gi, ' ')
      .split(/\s+/)
      .filter((token) => token.length > 2 && !CORPORATE_SUFFIXES.has(token))
  )
}

/**
 * Dos razones sociales se consideran el mismo proveedor si comparten alguna
 * palabra distintiva. "Northwind Logistics Inc." y "Northwind Logistics"
 * coinciden; "Quantum Freight Systems" y "Northwind Logistics" no.
 */
function sameVendor(a: string, b: string): boolean {
  const tokensA = vendorTokens(a)
  const tokensB = vendorTokens(b)
  if (tokensA.size === 0 || tokensB.size === 0) return false
  for (const token of tokensA) if (tokensB.has(token)) return true
  return false
}

/** Normaliza la descripción de un ítem para poder compararla. */
function descriptionTokens(description: string): string[] {
  return description
    .toLowerCase()
    .replace(/[^a-z0-9áéíóúñü\s]/gi, ' ')
    .split(/\s+/)
    .filter((token) => token.length > 1)
}

/**
 * Empareja dos descripciones de ítem por solapamiento de palabras.
 *
 * El OCR y el modelo introducen variaciones menores ("Circular saw blade
 * 190mm" contra "Circular saw blade, 190 mm"), así que una comparación exacta
 * produciría faltantes falsos. Se exige que la mitad de las palabras de la
 * descripción más corta aparezcan en la otra.
 */
function sameItem(a: string, b: string): boolean {
  const tokensA = descriptionTokens(a)
  const tokensB = descriptionTokens(b)
  if (tokensA.length === 0 || tokensB.length === 0) return false

  const setB = new Set(tokensB)
  const shared = tokensA.filter((token) => setB.has(token)).length
  return shared / Math.min(tokensA.length, tokensB.length) >= 0.5
}

const money = (value: number) => value.toFixed(2)

/**
 * Verifica el veredicto del modelo contra la evidencia, de forma determinista.
 *
 * El modelo transcribe; el código compara. Esa división no es arbitraria: en
 * las corridas contra los datos de prueba, el modelo leyó bien los valores de
 * ambos documentos y aun así declaró coincidencias que no existían. Transcribir
 * texto de un OCR es lo que un LLM hace bien; restar dos números y cruzar dos
 * listas es lo que hace bien el código, y además es exacto y auditable.
 *
 * Las correcciones van siempre en dirección conservadora: una coincidencia
 * puede degradarse a discrepancia o a incertidumbre, nunca al revés.
 */
export function verifyAgainstEvidence(
  invoice: InvoiceData,
  audit: AuditResult,
  retrievalScore: number | null = null
): AuditResult {
  const uncertain = (summary: string, confidence = 0.4): AuditResult => ({
    ...audit,
    verdict: 'UNCERTAIN',
    confidence: Math.min(audit.confidence, confidence),
    summary,
    discrepancies: []
  })

  // --- ¿La evidencia es siquiera de esta factura? -------------------------
  // Se verifica antes que nada: reportar una diferencia contra el documento
  // equivocado es una acusación falsa, peor que no decir nada.

  if (audit.supportDocumentId.trim().length === 0) {
    return uncertain(
      'No se identificó un documento de respaldo, así que no hay evidencia para sostener un veredicto.'
    )
  }

  const vendorKnown = audit.supportVendorName.trim().length > 0
  if (vendorKnown && !sameVendor(invoice.vendorName, audit.supportVendorName)) {
    return uncertain(
      `El respaldo recuperado (${audit.supportDocumentId}) es de ${audit.supportVendorName}, no de ${invoice.vendorName}; no corresponde a esta factura.`
    )
  }

  // Sin proveedor confirmado y con una recuperación floja, no hay nada sólido.
  if (!vendorKnown && retrievalScore !== null && retrievalScore < WEAK_RETRIEVAL_SCORE) {
    return uncertain(
      `No se pudo confirmar que ${audit.supportDocumentId} corresponda a esta factura (similitud ${retrievalScore.toFixed(2)}).`
    )
  }

  // Una referencia de PO explícita que no coincide con el respaldo recuperado
  // significa que se está comparando contra otra orden de compra.
  const poReference = invoice.poReference.trim()
  if (
    poReference.length > 0 &&
    !audit.supportDocumentId.toLowerCase().includes(poReference.toLowerCase())
  ) {
    return uncertain(
      `La factura referencia ${poReference} pero el respaldo recuperado es ${audit.supportDocumentId}; no se encontró la orden de compra citada.`
    )
  }

  if (audit.supportTotalAmount <= 0) {
    return uncertain(
      `No se pudo leer el total de ${audit.supportDocumentId}; no hay forma de confirmar que la factura coincida.`
    )
  }

  // --- Comparación determinista -------------------------------------------

  const discrepancies: DiscrepancyReport[] = []
  const headline: string[] = []

  // 1. Totales.
  const delta = invoice.totalAmount - audit.supportTotalAmount
  if (Math.abs(delta) > AMOUNT_EPSILON) {
    discrepancies.push({
      field: 'total',
      invoiceValue: `${invoice.currency} ${money(invoice.totalAmount)}`,
      supportValue: `${invoice.currency} ${money(audit.supportTotalAmount)}`,
      difference: `La factura ${delta > 0 ? 'excede' : 'queda por debajo'} del respaldo en ${money(Math.abs(delta))} ${invoice.currency}.`
    })
    headline.push(`el total difiere en ${money(Math.abs(delta))} ${invoice.currency}`)
  }

  // 2. Coherencia interna de la factura: los ítems deben sumar el total.
  //    Una entrega parcial facturada por el monto completo se delata acá sola,
  //    sin necesidad de mirar el respaldo.
  if (invoice.items.length > 0) {
    const itemsSum = invoice.items.reduce((sum, item) => sum + item.amount, 0)
    const internalDelta = invoice.totalAmount - itemsSum
    if (Math.abs(internalDelta) > AMOUNT_EPSILON) {
      discrepancies.push({
        field: 'coherencia interna',
        invoiceValue: `total ${money(invoice.totalAmount)}`,
        supportValue: `ítems suman ${money(itemsSum)}`,
        difference: `La factura cobra ${money(Math.abs(internalDelta))} ${invoice.currency} ${internalDelta > 0 ? 'más' : 'menos'} de lo que detalla en sus ítems.`
      })
      headline.push(`los ítems no suman el total facturado`)
    }
  }

  // 3. Cruce de listas de ítems.
  const unmatchedInvoice = [...invoice.items]
  const onlyOnSupport: string[] = []

  for (const supportItem of audit.supportItems) {
    const index = unmatchedInvoice.findIndex((item) =>
      sameItem(item.description, supportItem.description)
    )

    if (index === -1) {
      onlyOnSupport.push(supportItem.description)
      discrepancies.push({
        field: `ítem no facturado: ${supportItem.description}`,
        invoiceValue: 'ausente',
        supportValue: `${supportItem.quantity} × ${money(supportItem.unitPrice)}`,
        difference: 'El respaldo autoriza este ítem pero la factura no lo detalla.'
      })
      continue
    }

    const invoiceItem = unmatchedInvoice[index]!
    unmatchedInvoice.splice(index, 1)

    if (Math.abs(invoiceItem.unitPrice - supportItem.unitPrice) > AMOUNT_EPSILON) {
      discrepancies.push({
        field: `precio unitario: ${invoiceItem.description}`,
        invoiceValue: money(invoiceItem.unitPrice),
        supportValue: money(supportItem.unitPrice),
        difference: `Se factura a ${money(invoiceItem.unitPrice)} un ítem autorizado a ${money(supportItem.unitPrice)}.`
      })
    }

    if (Math.abs(invoiceItem.quantity - supportItem.quantity) > AMOUNT_EPSILON) {
      discrepancies.push({
        field: `cantidad: ${invoiceItem.description}`,
        invoiceValue: String(invoiceItem.quantity),
        supportValue: String(supportItem.quantity),
        difference: `Se facturan ${invoiceItem.quantity} unidades contra ${supportItem.quantity} autorizadas.`
      })
    }
  }

  for (const extra of unmatchedInvoice) {
    discrepancies.push({
      field: `ítem no autorizado: ${extra.description}`,
      invoiceValue: `${extra.quantity} × ${money(extra.unitPrice)}`,
      supportValue: 'ausente',
      difference: 'Se factura un ítem que el respaldo no autoriza.'
    })
  }

  const itemIssues = onlyOnSupport.length + unmatchedInvoice.length
  if (itemIssues > 0) {
    headline.push(`${itemIssues} ítem(s) no se corresponden`)
  }

  if (discrepancies.length === 0) {
    // La evidencia respalda una coincidencia. Se conserva el resumen del
    // modelo, que suele redactarlo mejor que una plantilla.
    return { ...audit, verdict: 'MATCH', discrepancies: [] }
  }

  return {
    ...audit,
    verdict: 'DISCREPANCY',
    confidence: 0.99,
    summary: `Contra ${audit.supportDocumentId}: ${headline.join(' y ')}.`,
    discrepancies
  }
}

/**
 * Recupera el nombre de archivo que se antepuso al contenido durante la
 * indexación, para poder nombrar el respaldo en el reporte.
 */
export function supportLabel(chunkContent: string): string | null {
  const match = /^\s*\[([^\]]+)\]/.exec(chunkContent)
  return match?.[1] ?? null
}

/** Recorta el contexto para que la ventana del modelo no se desborde. */
function clip(text: string, max: number): string {
  return text.length <= max ? text : `${text.slice(0, max)}\n[...texto truncado...]`
}

export interface AuditInput {
  ocr: OcrDocumentResult
  hits: RagSearchResult[]
}

/**
 * Extrae y audita todas las facturas con un único LLM cargado.
 *
 * Cada factura se resuelve de forma independiente: un fallo de extracción o de
 * auditoría degrada esa fila a `ERROR` o `UNCERTAIN` y el lote continúa.
 */
export async function extractAndAudit(
  inputs: AuditInput[],
  onProgress: OnProgress = noop
): Promise<ReconciliationVerdict[]> {
  if (inputs.length === 0) return []

  return withModel(
    QWEN3_4B_INST_Q4_K_M.name,
    (onDownload) =>
      loadModel({
        modelSrc: QWEN3_4B_INST_Q4_K_M,
        modelConfig: { ctx_size: 8192 },
        onProgress: onDownload
      }),
    onProgress,
    async (modelId) => {
      const verdicts: ReconciliationVerdict[] = []

      for (const [index, input] of inputs.entries()) {
        const name = path.basename(input.ocr.file)
        const startedAll = now()
        const timings: StageTiming[] = [{ stage: 'ocr', ms: input.ocr.ms }]

        // El OCR ya falló: no hay nada que extraer.
        if (input.ocr.error !== null || input.ocr.text.length === 0) {
          verdicts.push({
            file: name,
            status: 'ERROR',
            invoice: null,
            audit: null,
            matchedSupportDoc: null,
            supportScore: null,
            error: input.ocr.error ?? 'El OCR no devolvió texto.',
            timings,
            totalMs: now() - startedAll
          })
          continue
        }

        // --- Extracción -----------------------------------------------------
        onProgress({
          stage: 'extract',
          message: `Extrayendo ${name}`,
          current: index + 1,
          total: inputs.length
        })

        const extractStarted = now()
        const extraction = await completeJson<InvoiceData>(
          modelId,
          [
            { role: 'system', content: EXTRACTION_SYSTEM },
            {
              role: 'user',
              content: `Texto OCR de la factura:\n\n${clip(input.ocr.text, 6000)}`
            }
          ],
          INVOICE_JSON_SCHEMA,
          'invoice_data',
          InvoiceDataSchema
        )
        timings.push({ stage: 'extract', ms: now() - extractStarted })

        if (!extraction.ok) {
          verdicts.push({
            file: name,
            status: 'ERROR',
            invoice: null,
            audit: null,
            matchedSupportDoc: null,
            supportScore: null,
            error: `Falló la extracción estructurada: ${extraction.error}`,
            timings,
            totalMs: now() - startedAll
          })
          continue
        }

        const invoice = extraction.data
        const best = input.hits[0] ?? null
        // El id que devuelve la búsqueda es un UUID interno, inservible para un
        // auditor. El nombre real del archivo viaja como prefijo del contenido,
        // que es justamente para lo que se antepuso al indexar.
        const bestLabel = best !== null ? (supportLabel(best.content) ?? best.id) : null
        const supportContext =
          input.hits.length > 0
            ? input.hits
                .map((hit, i) => `--- Respaldo ${i + 1} (score ${hit.score.toFixed(3)}) ---\n${hit.content}`)
                .join('\n\n')
            : 'NO SE RECUPERÓ NINGÚN DOCUMENTO DE RESPALDO.'

        // --- Auditoría ------------------------------------------------------
        onProgress({
          stage: 'audit',
          message: `Auditando ${name}`,
          current: index + 1,
          total: inputs.length
        })

        const auditStarted = now()
        const audit = await completeJson<AuditResult>(
          modelId,
          [
            { role: 'system', content: AUDIT_SYSTEM },
            {
              role: 'user',
              content: [
                'FACTURA EXTRAÍDA (JSON):',
                JSON.stringify(invoice, null, 2),
                '',
                'DOCUMENTOS DE RESPALDO RECUPERADOS:',
                clip(supportContext, 6000),
                '',
                'Emití tu veredicto en JSON.'
              ].join('\n')
            }
          ],
          AUDIT_JSON_SCHEMA,
          'audit_result',
          AuditResultSchema
        )
        timings.push({ stage: 'audit', ms: now() - auditStarted })

        if (!audit.ok) {
          // Falló la auditoría, pero la extracción sirvió: se conserva lo
          // obtenido y se marca UNCERTAIN con el motivo, en vez de perderlo todo.
          verdicts.push({
            file: name,
            status: 'OK',
            invoice,
            audit: {
              supportDocumentId: bestLabel ?? '',
              supportVendorName: '',
              supportTotalAmount: 0,
              supportItems: [],
              discrepancies: [],
              verdict: 'UNCERTAIN',
              confidence: 0,
              summary: `El modelo de auditoría no produjo un veredicto válido: ${audit.error}`
            },
            matchedSupportDoc: bestLabel,
            supportScore: best?.score ?? null,
            error: audit.error,
            timings,
            totalMs: now() - startedAll
          })
          continue
        }

        verdicts.push({
          file: name,
          status: 'OK',
          invoice,
          audit: verifyAgainstEvidence(invoice, audit.data, best?.score ?? null),
          matchedSupportDoc: bestLabel,
          supportScore: best?.score ?? null,
          error: null,
          timings,
          totalMs: now() - startedAll
        })
      }

      return verdicts
    }
  )
}

// ---------------------------------------------------------------------------
// Orquestación
// ---------------------------------------------------------------------------

export interface ReconcileOptions {
  invoicesDir: string
  supportDir: string
  workspace?: string
  onProgress?: OnProgress
}

/**
 * Corre el pipeline completo: OCR → indexación y recuperación → extracción y
 * auditoría. Las tres fases cargan un solo modelo por vez.
 */
export async function reconcileFolders(options: ReconcileOptions): Promise<ReconciliationVerdict[]> {
  const onProgress = options.onProgress ?? noop
  const workspace = options.workspace ?? `invoices-${Date.now()}`

  const [invoiceFiles, supportFiles] = await Promise.all([
    listDocuments(options.invoicesDir),
    listDocuments(options.supportDir)
  ])

  if (invoiceFiles.length === 0) {
    throw new Error(`No se encontraron facturas soportadas en ${options.invoicesDir}`)
  }

  // Fase 1: un solo modelo OCR para facturas y respaldos.
  onProgress({ stage: 'ocr', message: 'Fase 1/3 — OCR de todos los documentos' })
  const allOcr = await ocrDocuments([...supportFiles, ...invoiceFiles], onProgress)
  const supportOcr = allOcr.slice(0, supportFiles.length)
  const invoiceOcr = allOcr.slice(supportFiles.length)

  // Fase 2: embeddings + recuperación, y el modelo se descarga al salir.
  onProgress({ stage: 'index', message: 'Fase 2/3 — Indexación RAG y recuperación' })
  let retrieved = new Map<string, RagSearchResult[]>()
  try {
    const result = await indexSupportDocuments(supportOcr, invoiceOcr, workspace, onProgress)
    retrieved = result.retrieved
  } catch (error) {
    // Sin RAG el pipeline sigue: las facturas quedan sin evidencia y la
    // auditoría las marcará UNCERTAIN, que es el comportamiento honesto.
    onProgress({
      stage: 'index',
      message: `La indexación RAG falló (${error instanceof Error ? error.message : String(error)}); se continúa sin evidencia de respaldo.`
    })
  }

  // Fase 3: el LLM entra en memoria recién ahora.
  onProgress({ stage: 'extract', message: 'Fase 3/3 — Extracción estructurada y auditoría' })
  const verdicts = await extractAndAudit(
    invoiceOcr.map((entry) => ({ ocr: entry, hits: retrieved.get(entry.file) ?? [] })),
    onProgress
  )

  await ragCloseWorkspace({ workspace, deleteOnClose: true }).catch(() => {})
  onProgress({ stage: 'done', message: 'Pipeline completo.' })

  return verdicts
}

/** Cierra la conexión RPC del SDK para que el proceso pueda terminar. */
export async function shutdown(): Promise<void> {
  await close().catch(() => {})
}
