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
  type ReasonCode,
  type ReconciliationVerdict,
  type StageTiming,
  type VerifiedAudit
} from '../types.js'
import { missingCriticalFields, normalizeEvidence, normalizeInvoice } from './normalize.js'
import { isSupportedDocument, toImagePages } from './rasterize.js'
import { resolvePoEvidence } from './retrieval.js'

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
4. "supportCurrency": el código de moneda impreso en el respaldo (USD, EUR, ARS, etc.), transcrito del texto del respaldo. Si el respaldo no menciona ninguna moneda legible, usá "". NUNCA lo copies de la factura: si el respaldo está en otra moneda, ese dato es exactamente lo que el verificador necesita saber.
5. "supportItems": TODOS los ítems que figuran en el respaldo, uno por uno, con su descripción, cantidad, precio unitario e importe. Transcribí la lista completa aunque sea larga: si omitís un ítem, el verificador va a creer que la factura dejó de facturarlo.
6. "discrepancies": dejalo vacío ([]). Lo completa el verificador.
7. "verdict", 8. "confidence", 9. "summary": tu impresión general. El verificador puede corregirla.

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

/**
 * Forma normalizada de una razón social: minúsculas, sin puntuación, sin
 * sufijos societarios y con el espaciado colapsado. Los puntos y apóstrofos
 * se eliminan (no se reemplazan por espacio) para que "S.A." colapse en "sa"
 * y quede cubierto por CORPORATE_SUFFIXES.
 */
function normalizedVendorName(name: string): string {
  return name
    .toLowerCase()
    .replace(/[.'’]/g, '')
    .replace(/[^a-z0-9áéíóúñü\s]/gi, ' ')
    .split(/\s+/)
    .filter((token) => token.length > 0 && !CORPORATE_SUFFIXES.has(token))
    .join(' ')
}

/** Reduce una razón social a sus palabras distintivas. */
function vendorTokens(name: string): Set<string> {
  return new Set(
    normalizedVendorName(name)
      .split(' ')
      .filter((token) => token.length > 2)
  )
}

/**
 * Decide si dos razones sociales nombran al mismo proveedor, con una
 * jerarquía conservadora:
 *
 *   1. Igualdad exacta tras normalizar (minúsculas, puntuación, espaciado,
 *      sufijos societarios): "ACME S.A." ≡ "acme sa".
 *   2. Solapamiento de tokens distintivos: la MAYORÍA de los tokens del
 *      nombre más corto debe aparecer en el otro (>= 2 compartidos, o >= 60%).
 *      "Northwind Logistics Inc." ≡ "Northwind Logistics"; un token garbleado
 *      por OCR en un nombre de tres palabras sigue coincidiendo.
 *   3. Cualquier cosa más débil se trata como proveedores DISTINTOS, lo que
 *      vía la salvaguarda existente degrada a UNCERTAIN — nunca a una
 *      acusación por identidad de proveedor.
 *
 * Compartir UNA sola palabra no prueba identidad: "Acme Logistics" y "Beta
 * Logistics" son empresas distintas que comparten un rubro. La similitud
 * difusa sola jamás prueba identidad.
 */
export function sameVendor(a: string, b: string): boolean {
  const normA = normalizedVendorName(a)
  const normB = normalizedVendorName(b)
  if (normA.length === 0 || normB.length === 0) return false
  if (normA === normB) return true

  const tokensA = vendorTokens(a)
  const tokensB = vendorTokens(b)
  if (tokensA.size === 0 || tokensB.size === 0) return false

  const [shorter, longer] = tokensA.size <= tokensB.size ? [tokensA, tokensB] : [tokensB, tokensA]
  let shared = 0
  for (const token of shorter) if (longer.has(token)) shared++

  return shared >= 2 || shared / shorter.size >= 0.6
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
  rawInvoice: InvoiceData,
  rawAudit: AuditResult,
  retrievalScore: number | null = null
): VerifiedAudit {
  // El verificador consume ÚNICAMENTE datos normalizados: los centinelas de
  // extracción ("" y 0) ya llegaron convertidos en `null` explícito, así que
  // acá "faltante" y "valor real" no pueden confundirse.
  const invoice = normalizeInvoice(rawInvoice)
  const evidence = normalizeEvidence(rawAudit)

  const uncertain = (reasonCode: ReasonCode, summary: string, confidence = 0.4): VerifiedAudit => ({
    ...rawAudit,
    verdict: 'UNCERTAIN',
    confidence: Math.min(rawAudit.confidence, confidence),
    summary,
    discrepancies: [],
    reasonCode
  })

  // --- ¿La factura es siquiera auditable? ---------------------------------
  // Sin identificador, proveedor, total o moneda de la FACTURA no hay
  // comparación posible. Faltantes no críticos (fecha, PO) no bloquean.
  const missing = missingCriticalFields(invoice)
  if (
    invoice.invoiceNumber === null ||
    invoice.vendorName === null ||
    invoice.totalAmount === null ||
    invoice.currency === null
  ) {
    return uncertain(
      'MISSING_CRITICAL_FIELD',
      `No se pudo leer ${missing.join(', ')} de la factura; sin esos campos no hay comparación posible.`
    )
  }

  // --- ¿La evidencia es siquiera de esta factura? -------------------------
  // Se verifica antes que nada: reportar una diferencia contra el documento
  // equivocado es una acusación falsa, peor que no decir nada.

  if (evidence.supportDocumentId === null) {
    return uncertain(
      'NO_EVIDENCE',
      'No se identificó un documento de respaldo, así que no hay evidencia para sostener un veredicto.'
    )
  }
  const supportDocumentId = evidence.supportDocumentId

  if (
    evidence.supportVendorName !== null &&
    !sameVendor(invoice.vendorName, evidence.supportVendorName)
  ) {
    return uncertain(
      'VENDOR_MISMATCH',
      `El respaldo recuperado (${supportDocumentId}) es de ${evidence.supportVendorName}, no de ${invoice.vendorName}; no corresponde a esta factura.`
    )
  }

  // Sin proveedor confirmado y con una recuperación floja, no hay nada sólido.
  if (
    evidence.supportVendorName === null &&
    retrievalScore !== null &&
    retrievalScore < WEAK_RETRIEVAL_SCORE
  ) {
    return uncertain(
      'WEAK_RETRIEVAL',
      `No se pudo confirmar que ${supportDocumentId} corresponda a esta factura (similitud ${retrievalScore.toFixed(2)}).`
    )
  }

  // Una referencia de PO explícita que no coincide con el respaldo recuperado
  // significa que se está comparando contra otra orden de compra.
  if (
    invoice.poReference !== null &&
    !supportDocumentId.toLowerCase().includes(invoice.poReference.toLowerCase())
  ) {
    return uncertain(
      'PO_NOT_FOUND',
      `La factura referencia ${invoice.poReference} pero el respaldo recuperado es ${supportDocumentId}; no se encontró la orden de compra citada.`
    )
  }

  // --- ¿Los montos son siquiera comparables? ------------------------------
  // Dos monedas distintas hacen incomparables los montos: 1000 ARS contra
  // 1000 USD NO es una coincidencia. Nunca se convierte implícitamente; una
  // conversión necesita un tipo de cambio y una fecha que este sistema no
  // tiene. La comparación es igualdad de códigos normalizados — no depende
  // de ninguna lista de monedas soportadas.
  const currency = invoice.currency
  const currencyNotes: string[] = []

  if (evidence.supportCurrency !== null && evidence.supportCurrency !== currency) {
    return uncertain(
      'CURRENCY_MISMATCH',
      `La factura está en ${currency} pero ${supportDocumentId} está en ${evidence.supportCurrency}; los montos no son comparables sin una conversión explícita.`
    )
  }
  if (evidence.supportCurrency === null) {
    // Sin moneda legible en el respaldo, los chequeos numéricos siguen (los
    // montos podrían igualmente delatar un desvío), pero el resultado lo dice:
    // no se fabrica una moneda que el documento no muestra.
    currencyNotes.push('moneda del respaldo sin verificar')
  }

  if (evidence.supportTotalAmount === null) {
    return uncertain(
      'EVIDENCE_UNREADABLE',
      `No se pudo leer el total de ${supportDocumentId}; no hay forma de confirmar que la factura coincida.`
    )
  }

  // --- Comparación determinista -------------------------------------------

  const discrepancies: DiscrepancyReport[] = []
  const headline: string[] = []

  // 1. Totales.
  const delta = invoice.totalAmount - evidence.supportTotalAmount
  if (Math.abs(delta) > AMOUNT_EPSILON) {
    discrepancies.push({
      reasonCode: 'TOTAL_MISMATCH',
      field: 'total',
      invoiceValue: `${currency} ${money(invoice.totalAmount)}`,
      supportValue: `${currency} ${money(evidence.supportTotalAmount)}`,
      difference: `La factura ${delta > 0 ? 'excede' : 'queda por debajo'} del respaldo en ${money(Math.abs(delta))} ${currency}.`
    })
    headline.push(`el total difiere en ${money(Math.abs(delta))} ${currency}`)
  }

  // 2. Coherencia interna de la factura: los ítems deben sumar el total.
  //    Una entrega parcial facturada por el monto completo se delata acá sola,
  //    sin necesidad de mirar el respaldo.
  //
  //    Pero este chequeo compara dos números que salen del MISMO OCR, así que
  //    un renglón mal leído produciría una acusación falsa. Sólo se corre si
  //    cada renglón es internamente coherente — importe igual a cantidad por
  //    precio unitario. Si algún renglón no cierra, el que no es confiable es
  //    el OCR, no la factura, y no se acusa a nadie.
  const linesAreTrustworthy = invoice.items.every(
    (item) => Math.abs(item.amount - item.quantity * item.unitPrice) <= AMOUNT_EPSILON
  )

  if (invoice.items.length > 0 && linesAreTrustworthy) {
    const itemsSum = invoice.items.reduce((sum, item) => sum + item.amount, 0)
    const internalDelta = invoice.totalAmount - itemsSum
    if (Math.abs(internalDelta) > AMOUNT_EPSILON) {
      discrepancies.push({
        reasonCode: 'INTERNAL_SUM_MISMATCH',
        field: 'coherencia interna',
        invoiceValue: `total ${money(invoice.totalAmount)}`,
        supportValue: `ítems suman ${money(itemsSum)}`,
        difference: `La factura cobra ${money(Math.abs(internalDelta))} ${currency} ${internalDelta > 0 ? 'más' : 'menos'} de lo que detalla en sus ítems.`
      })
      headline.push('los ítems no suman el total facturado')
    }
  }

  // 3. Cruce de listas de ítems.
  const unmatchedInvoice = [...invoice.items]
  const onlyOnSupport: string[] = []

  for (const supportItem of evidence.supportItems) {
    const index = unmatchedInvoice.findIndex((item) =>
      sameItem(item.description, supportItem.description)
    )

    if (index === -1) {
      onlyOnSupport.push(supportItem.description)
      discrepancies.push({
        reasonCode: 'ITEM_NOT_ON_INVOICE',
        field: `ítem no facturado: ${supportItem.description}`,
        invoiceValue: 'ausente',
        supportValue: `${currency} ${money(supportItem.amount)}`,
        difference: 'El respaldo autoriza este ítem pero la factura no lo detalla.'
      })
      continue
    }

    const invoiceItem = unmatchedInvoice[index]!
    unmatchedInvoice.splice(index, 1)

    if (Math.abs(invoiceItem.unitPrice - supportItem.unitPrice) > AMOUNT_EPSILON) {
      discrepancies.push({
        reasonCode: 'UNIT_PRICE_MISMATCH',
        field: `precio unitario: ${invoiceItem.description}`,
        invoiceValue: money(invoiceItem.unitPrice),
        supportValue: money(supportItem.unitPrice),
        difference: `Se factura a ${money(invoiceItem.unitPrice)} un ítem autorizado a ${money(supportItem.unitPrice)}.`
      })
    }

    if (Math.abs(invoiceItem.quantity - supportItem.quantity) > AMOUNT_EPSILON) {
      discrepancies.push({
        reasonCode: 'QUANTITY_MISMATCH',
        field: `cantidad: ${invoiceItem.description}`,
        invoiceValue: String(invoiceItem.quantity),
        supportValue: String(supportItem.quantity),
        difference: `Se facturan ${invoiceItem.quantity} unidades contra ${supportItem.quantity} autorizadas.`
      })
    }
  }

  for (const extra of unmatchedInvoice) {
    discrepancies.push({
      reasonCode: 'ITEM_NOT_AUTHORIZED',
      field: `ítem no autorizado: ${extra.description}`,
      invoiceValue: `${currency} ${money(extra.amount)}`,
      supportValue: 'ausente',
      difference: 'Se factura un ítem que el respaldo no autoriza.'
    })
  }

  const itemIssues = onlyOnSupport.length + unmatchedInvoice.length
  if (itemIssues > 0) {
    headline.push(`${itemIssues} ítem(s) no se corresponden`)
  }

  const caveat = currencyNotes.length > 0 ? ` (${currencyNotes.join('; ')})` : ''

  if (discrepancies.length === 0) {
    // La evidencia respalda una coincidencia. Se conserva el resumen del
    // modelo, que suele redactarlo mejor que una plantilla.
    return {
      ...rawAudit,
      verdict: 'MATCH',
      discrepancies: [],
      summary: `${rawAudit.summary}${caveat}`,
      reasonCode: null
    }
  }

  return {
    ...rawAudit,
    verdict: 'DISCREPANCY',
    confidence: 0.99,
    summary: `Contra ${supportDocumentId}: ${headline.join(' y ')}.${caveat}`,
    discrepancies,
    // El motivo principal es el de la primera discrepancia encontrada; el
    // detalle completo viaja en cada entrada de `discrepancies`.
    reasonCode: discrepancies[0]!.reasonCode
  }
}

/**
 * Registra el número de factura de `file` en el mapa de vistos del lote y
 * devuelve el archivo donde ese número YA había aparecido, o `null` si es la
 * primera vez. Un número ilegible ("" tras normalizar) nunca se registra:
 * dos facturas ilegibles no son duplicados entre sí, son dos incógnitas.
 */
export function registerInvoiceNumber(
  seen: Map<string, string>,
  invoiceNumber: string,
  file: string
): string | null {
  const key = invoiceNumber.trim().toUpperCase()
  if (key.length === 0) return null

  const priorFile = seen.get(key)
  if (priorFile !== undefined) return priorFile

  seen.set(key, file)
  return null
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

/**
 * De dónde salió la evidencia de una factura. `exact` lleva el texto OCR
 * completo del respaldo (no un fragmento RAG): la coincidencia de PO ya
 * identificó el documento, y el auditor debe verlo entero. `po-not-found`
 * significa que la factura cita una orden que NO existe en el conjunto de
 * respaldos: no hay nada que auditar y el veredicto se decide sin modelo.
 */
export type EvidenceSource =
  | { kind: 'exact'; file: string; text: string; citedPo: string }
  | { kind: 'semantic'; hits: RagSearchResult[] }
  | { kind: 'po-not-found'; citedPo: string }

export interface AuditInput {
  ocr: OcrDocumentResult
  evidence: EvidenceSource
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
      // Número de factura ya visto en este lote → archivo donde apareció.
      // La detección de duplicados es código puro sobre lo ya extraído.
      const seenInvoiceNumbers = new Map<string, string>()

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

        // Duplicado dentro del lote: el mismo número de factura ya apareció en
        // un archivo anterior. Es una discrepancia establecida por código puro
        // (una factura presentada dos veces se paga dos veces); se marca la
        // ocurrencia posterior, referenciando al archivo original, sin llamar
        // al modelo de auditoría.
        const duplicateOf = registerInvoiceNumber(seenInvoiceNumbers, invoice.invoiceNumber, name)
        if (duplicateOf !== null) {
          verdicts.push({
            file: name,
            status: 'OK',
            invoice,
            audit: {
              supportDocumentId: duplicateOf,
              supportVendorName: '',
              supportTotalAmount: 0,
              supportCurrency: '',
              supportItems: [],
              discrepancies: [
                {
                  reasonCode: 'DUPLICATE_INVOICE',
                  field: 'invoiceNumber',
                  invoiceValue: invoice.invoiceNumber,
                  supportValue: `ya presentada en ${duplicateOf}`,
                  difference: `El número ${invoice.invoiceNumber} ya apareció en ${duplicateOf} dentro de este mismo lote; una factura presentada dos veces se paga dos veces.`
                }
              ],
              verdict: 'DISCREPANCY',
              confidence: 0.99,
              summary: `Factura duplicada: ${invoice.invoiceNumber} ya fue presentada en ${duplicateOf}.`,
              reasonCode: 'DUPLICATE_INVOICE'
            },
            matchedSupportDoc: duplicateOf,
            supportScore: null,
            error: null,
            timings,
            totalMs: now() - startedAll
          })
          continue
        }

        // La factura cita una orden de compra que NO existe en el conjunto de
        // respaldos. Eso es un hecho establecido por búsqueda exacta, no una
        // impresión: el veredicto se decide acá, sin llamar al modelo. Caer a
        // la búsqueda semántica "a ver si hay algo parecido" sólo puede traer
        // un PO ajeno y fabricar acusaciones falsas.
        if (input.evidence.kind === 'po-not-found') {
          verdicts.push({
            file: name,
            status: 'OK',
            invoice,
            audit: {
              supportDocumentId: '',
              supportVendorName: '',
              supportTotalAmount: 0,
              supportCurrency: '',
              supportItems: [],
              discrepancies: [],
              verdict: 'UNCERTAIN',
              confidence: 0.3,
              summary: `La factura cita ${input.evidence.citedPo} pero esa orden no aparece en ningún documento de respaldo; la factura puede carecer de respaldo real.`,
              reasonCode: 'PO_NOT_FOUND'
            },
            matchedSupportDoc: null,
            supportScore: null,
            error: null,
            timings,
            totalMs: now() - startedAll
          })
          continue
        }

        let bestLabel: string | null
        let supportScore: number | null
        let supportContext: string

        if (input.evidence.kind === 'exact') {
          // Evidencia resuelta por PO exacto: el documento entero, no un
          // fragmento. El score semántico no aplica (no hubo búsqueda).
          bestLabel = path.basename(input.evidence.file)
          supportScore = null
          supportContext = `--- Respaldo (coincidencia exacta de PO ${input.evidence.citedPo}) ---\n[${bestLabel}]\n${input.evidence.text}`
        } else {
          const hits = input.evidence.hits
          const best = hits[0] ?? null
          // El id que devuelve la búsqueda es un UUID interno, inservible para
          // un auditor. El nombre real del archivo viaja como prefijo del
          // contenido, que es justamente para lo que se antepuso al indexar.
          bestLabel = best !== null ? (supportLabel(best.content) ?? best.id) : null
          supportScore = best?.score ?? null
          supportContext =
            hits.length > 0
              ? hits
                  .map((hit, i) => `--- Respaldo ${i + 1} (score ${hit.score.toFixed(3)}) ---\n${hit.content}`)
                  .join('\n\n')
              : 'NO SE RECUPERÓ NINGÚN DOCUMENTO DE RESPALDO.'
        }

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
              supportCurrency: '',
              supportItems: [],
              discrepancies: [],
              verdict: 'UNCERTAIN',
              confidence: 0,
              summary: `El modelo de auditoría no produjo un veredicto válido: ${audit.error}`,
              reasonCode: 'MODEL_OUTPUT_INVALID'
            },
            matchedSupportDoc: bestLabel,
            supportScore,
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
          audit: verifyAgainstEvidence(invoice, audit.data, supportScore),
          matchedSupportDoc: bestLabel,
          supportScore,
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

  // Fase 2a: resolución EXACTA de PO. Es puro trabajo de strings sobre los
  // resultados de OCR — corre sin ningún modelo cargado, así que no le cuesta
  // nada al presupuesto de memoria. Una factura que cita un PO se resuelve por
  // identificador; la búsqueda semántica queda sólo para las que no citan
  // ninguno. La estrategia completa está documentada en retrieval.ts.
  onProgress({ stage: 'index', message: 'Fase 2/3 — Resolución de respaldos (PO exacto primero)' })
  const lookupDocs = supportOcr
    .filter((d) => d.error === null && d.text.length > 0)
    .map((d) => ({ file: d.file, text: d.text }))
  const evidenceByFile = new Map<string, EvidenceSource>()
  const semanticOcr: OcrDocumentResult[] = []

  for (const inv of invoiceOcr) {
    if (inv.error !== null || inv.text.length === 0) {
      evidenceByFile.set(inv.file, { kind: 'semantic', hits: [] })
      continue
    }
    const resolution = resolvePoEvidence(inv.text, lookupDocs)
    if (resolution.kind === 'exact') {
      onProgress({
        stage: 'index',
        message: `${path.basename(inv.file)} → ${path.basename(resolution.doc.file)} (PO exacto ${resolution.citedPo}).`
      })
      evidenceByFile.set(inv.file, {
        kind: 'exact',
        file: resolution.doc.file,
        text: resolution.doc.text,
        citedPo: resolution.citedPo
      })
    } else if (resolution.kind === 'not-found') {
      onProgress({
        stage: 'index',
        message: `${path.basename(inv.file)} cita ${resolution.citedPo}, ausente de los respaldos.`
      })
      evidenceByFile.set(inv.file, { kind: 'po-not-found', citedPo: resolution.citedPo })
    } else {
      semanticOcr.push(inv)
    }
  }

  // Fase 2b: embeddings + recuperación semántica, sólo si alguna factura no
  // citó PO. El modelo se descarga al salir.
  if (semanticOcr.length > 0) {
    try {
      const result = await indexSupportDocuments(supportOcr, semanticOcr, workspace, onProgress)
      for (const inv of semanticOcr) {
        evidenceByFile.set(inv.file, { kind: 'semantic', hits: result.retrieved.get(inv.file) ?? [] })
      }
    } catch (error) {
      // Sin RAG el pipeline sigue: las facturas quedan sin evidencia y la
      // auditoría las marcará UNCERTAIN, que es el comportamiento honesto.
      onProgress({
        stage: 'index',
        message: `La indexación RAG falló (${error instanceof Error ? error.message : String(error)}); se continúa sin evidencia de respaldo.`
      })
      for (const inv of semanticOcr) {
        if (!evidenceByFile.has(inv.file)) evidenceByFile.set(inv.file, { kind: 'semantic', hits: [] })
      }
    }
  } else {
    onProgress({
      stage: 'index',
      message: 'Todas las facturas se resolvieron por PO exacto; no hace falta búsqueda semántica.'
    })
  }

  // Fase 3: el LLM entra en memoria recién ahora.
  onProgress({ stage: 'extract', message: 'Fase 3/3 — Extracción estructurada y auditoría' })
  const verdicts = await extractAndAudit(
    invoiceOcr.map((entry) => ({
      ocr: entry,
      evidence: evidenceByFile.get(entry.file) ?? { kind: 'semantic', hits: [] }
    })),
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
