/**
 * Recuperación híbrida de órdenes de compra: exacto primero, semántica después.
 *
 * Cuando una factura cita explícitamente una orden de compra ("PO Reference:
 * PO-5003"), buscarla por similitud semántica es usar la herramienta equivocada:
 * un embedding puede traer con score alto una orden PARECIDA pero AJENA, y
 * comparar contra el documento equivocado produce acusaciones falsas. La cita
 * es un identificador, y los identificadores se resuelven por igualdad.
 *
 * Estrategia (implementada acá, en código puro, sin ningún modelo cargado):
 *   1. PO citado en la factura → búsqueda EXACTA (normalizada) contra los
 *      nombres de archivo de los respaldos y contra los números de PO que
 *      aparecen en su texto OCR. Si aparece, ESE documento es la evidencia y
 *      se pasa su texto OCR completo (no un fragmento RAG) a la auditoría.
 *   2. PO citado pero ausente de TODO el conjunto de respaldos → la orden
 *      citada genuinamente no está: el pipeline degrada a UNCERTAIN con
 *      PO_NOT_FOUND. Deliberadamente NO se cae a la búsqueda semántica "a ver
 *      si hay algo parecido": un PO parecido pero equivocado es peor que
 *      ningún PO.
 *   3. Sin PO citado → búsqueda semántica como siempre (topK 3), con las
 *      salvaguardas existentes de proveedor y WEAK_RETRIEVAL_SCORE.
 *
 * Todo lo de este módulo es trabajo de strings sobre resultados de OCR ya
 * calculados: respeta el presupuesto de memoria porque corre entre fases sin
 * ningún modelo en RAM.
 */
import path from 'node:path'

/** Documento de respaldo ya transcrito por OCR, listo para búsqueda exacta. */
export interface RetrievalDoc {
  file: string
  text: string
}

export type PoResolution =
  | { kind: 'exact'; doc: RetrievalDoc; citedPo: string }
  | { kind: 'not-found'; citedPo: string }
  | { kind: 'no-po' }

/**
 * Forma canónica de un identificador de PO: mayúsculas, sin separadores y sin
 * el prefijo "PO". "PO-5003", "po 5003", "PO5003" y "5003" coinciden entre sí.
 */
export function canonicalPo(raw: string): string {
  const normalized = raw.toUpperCase().replace(/[^A-Z0-9]/g, '')
  return normalized.startsWith('PO') ? normalized.slice(2) : normalized
}

/**
 * Token de PO en texto OCR: "PO", "P.O.", "PO#", "PO:" seguido de un
 * identificador que contiene al menos un dígito. La exigencia del dígito evita
 * capturar palabras ("PO Reference" no es un identificador; "PO-5003" sí).
 */
const PO_TOKEN = /\bP\.?O\.?[\s#:.-]{0,3}((?=[A-Z0-9-]*\d)[A-Z0-9][A-Z0-9-]{2,})/gi

/**
 * Orden de compra citada en el texto OCR de una factura, o `null` si el
 * documento no cita ninguna. Devuelve la primera cita encontrada, con su
 * texto visible (para mensajes) y su forma canónica (para comparar).
 */
export function findCitedPo(invoiceText: string): { display: string; canonical: string } | null {
  PO_TOKEN.lastIndex = 0
  const match = PO_TOKEN.exec(invoiceText)
  if (match === null) return null
  return { display: match[0].replace(/\s+/g, ' ').trim(), canonical: canonicalPo(match[1]!) }
}

/** Identificadores de PO que un documento de respaldo declara tener. */
export function poIdsOfSupportDoc(doc: RetrievalDoc): Set<string> {
  const ids = new Set<string>()

  // El nombre del archivo es la señal más fuerte: los respaldos se indexan
  // con su nombre como prefijo ([PO-5003.pdf]) y suele SER el número de PO.
  const base = path.basename(doc.file, path.extname(doc.file))
  const fromName = canonicalPo(base)
  if (/\d/.test(fromName)) ids.add(fromName)

  PO_TOKEN.lastIndex = 0
  for (const match of doc.text.matchAll(PO_TOKEN)) {
    ids.add(canonicalPo(match[1]!))
  }

  return ids
}

/**
 * Resuelve la evidencia de una factura por identificador exacto de PO.
 * Puro: no carga ningún modelo, no toca el disco.
 */
export function resolvePoEvidence(
  invoiceText: string,
  supportDocs: RetrievalDoc[]
): PoResolution {
  const cited = findCitedPo(invoiceText)
  if (cited === null || cited.canonical.length === 0) return { kind: 'no-po' }

  // Preferencia por la coincidencia de nombre de archivo; el texto OCR es el
  // segundo intento (cubre respaldos con nombre de archivo arbitrario).
  const byName = supportDocs.find((doc) => {
    const base = path.basename(doc.file, path.extname(doc.file))
    return canonicalPo(base) === cited.canonical
  })
  if (byName !== undefined) return { kind: 'exact', doc: byName, citedPo: cited.display }

  const byText = supportDocs.find((doc) => poIdsOfSupportDoc(doc).has(cited.canonical))
  if (byText !== undefined) return { kind: 'exact', doc: byText, citedPo: cited.display }

  return { kind: 'not-found', citedPo: cited.display }
}
