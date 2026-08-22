/**
 * Contratos de datos del reconciliador.
 *
 * Los schemas zod son la única fuente de verdad: de ellos se derivan tanto los
 * tipos de TypeScript como los JSON Schema que se le pasan al modelo vía
 * `responseFormat: { type: 'json_schema' }`. Así la gramática que restringe la
 * generación y la validación posterior no pueden divergir.
 */
import { z } from 'zod'

// ---------------------------------------------------------------------------
// Extracción de facturas
// ---------------------------------------------------------------------------

export const LineItemSchema = z.object({
  description: z.string(),
  quantity: z.number(),
  unitPrice: z.number(),
  amount: z.number()
})

export type LineItem = z.infer<typeof LineItemSchema>

export const InvoiceDataSchema = z.object({
  invoiceNumber: z.string(),
  vendorName: z.string(),
  /** Fecha tal como figura en el documento, preferentemente YYYY-MM-DD. */
  date: z.string(),
  totalAmount: z.number(),
  /** Código ISO de tres letras, p. ej. "USD". */
  currency: z.string(),
  /**
   * Orden de compra referenciada. Se usa cadena vacía en lugar de un campo
   * opcional: un campo siempre presente simplifica la gramática GBNF y evita
   * que el modelo omita la clave.
   */
  poReference: z.string(),
  items: z.array(LineItemSchema)
})

export type InvoiceData = z.infer<typeof InvoiceDataSchema>

// ---------------------------------------------------------------------------
// Auditoría y reconciliación
// ---------------------------------------------------------------------------

export const VerdictSchema = z.enum(['MATCH', 'DISCREPANCY', 'UNCERTAIN'])
export type Verdict = z.infer<typeof VerdictSchema>

/**
 * Motivos legibles por máquina. "¿Por qué el sistema decidió esto?" se
 * responde con estos códigos, sin volver a preguntarle al LLM. Los emite
 * exclusivamente el verificador determinista (el modelo deja `discrepancies`
 * vacío), así que cada código es trazable a una regla de código concreta.
 */
export const ReasonCodeSchema = z.enum([
  // Discrepancias numéricas y de ítems
  'TOTAL_MISMATCH',
  'INTERNAL_SUM_MISMATCH',
  'UNIT_PRICE_MISMATCH',
  'QUANTITY_MISMATCH',
  'ITEM_NOT_ON_INVOICE',
  'ITEM_NOT_AUTHORIZED',
  'DUPLICATE_INVOICE',
  // Motivos de incertidumbre
  'CURRENCY_MISMATCH',
  'PO_NOT_FOUND',
  'VENDOR_MISMATCH',
  'NO_EVIDENCE',
  'WEAK_RETRIEVAL',
  'EVIDENCE_UNREADABLE',
  'MISSING_CRITICAL_FIELD',
  'MODEL_OUTPUT_INVALID'
])

export type ReasonCode = z.infer<typeof ReasonCodeSchema>

export const DiscrepancyReportSchema = z.object({
  /** Motivo estructurado del desvío, emitido por el verificador determinista. */
  reasonCode: ReasonCodeSchema,
  /** Campo en conflicto: "totalAmount", "lineItem: Cordless drill 18V", etc. */
  field: z.string(),
  invoiceValue: z.string(),
  supportValue: z.string(),
  /** Explicación breve del desvío, legible por un auditor humano. */
  difference: z.string()
})

export type DiscrepancyReport = z.infer<typeof DiscrepancyReportSchema>

/**
 * Resultado de auditoría, **con la evidencia antes del veredicto**.
 *
 * El orden de las propiedades no es cosmético: llama.cpp compila el JSON Schema
 * a una gramática GBNF que emite las claves en el orden del schema, así que el
 * modelo está obligado a transcribir los valores del respaldo y a enumerar las
 * diferencias antes de poder escribir el token del veredicto. Con `verdict`
 * primero, el modelo se compromete a un juicio antes de haber mirado un solo
 * número — y ahí es donde alucina coincidencias.
 */
export const AuditResultSchema = z.object({
  /** Identificador del respaldo recuperado, p. ej. "PO-5003"; "" si no hay. */
  supportDocumentId: z.string(),
  /** Proveedor que figura en el respaldo, para poder detectar un cruce erróneo. */
  supportVendorName: z.string(),
  /** Total leído del documento de respaldo; 0 si no se pudo leer. */
  supportTotalAmount: z.number(),
  /**
   * Código de moneda impreso en el respaldo (ISO, p. ej. "USD"); "" si no se
   * pudo leer. Nunca se copia de la factura: si las monedas difieren, los
   * montos no son comparables y el veredicto debe degradar a UNCERTAIN.
   */
  supportCurrency: z.string(),
  /** Ítems transcritos del respaldo. El código los compara, no el modelo. */
  supportItems: z.array(LineItemSchema),
  discrepancies: z.array(DiscrepancyReportSchema),
  verdict: VerdictSchema,
  /** Confianza del modelo en su propio veredicto, de 0 a 1. */
  confidence: z.number(),
  /** Una sola frase: lo que un auditor necesita leer en menos de 5 segundos. */
  summary: z.string()
})

export type AuditResult = z.infer<typeof AuditResultSchema>

/**
 * Resultado de auditoría YA verificado por el código determinista. Además de
 * los códigos por discrepancia, lleva el motivo del veredicto en sí: los
 * UNCERTAIN también explican por qué, sin re-preguntarle al modelo. La
 * `confidence` del modelo sigue siendo cosmética: se muestra, pero jamás
 * participa de ninguna decisión.
 */
export type VerifiedAudit = AuditResult & {
  /** Motivo principal del veredicto; `null` cuando es MATCH. */
  reasonCode: ReasonCode | null
}

// ---------------------------------------------------------------------------
// Resultado del pipeline
// ---------------------------------------------------------------------------

/**
 * `OK` significa que el pipeline completó los pasos; el juicio de negocio vive
 * en `audit.verdict`. `ERROR` significa que un paso falló (OCR vacío, JSON
 * inválido, archivo ilegible) y el caso queda marcado para revisión manual.
 */
export type PipelineStatus = 'OK' | 'ERROR'

export interface StageTiming {
  stage: string
  ms: number
}

/**
 * Costo de una fase completa, con la carga del modelo separada del trabajo.
 *
 * La distinción es la que decide qué optimizar. Si el tiempo está en `loadMs`,
 * el problema es de E/S y la respuesta es cachear o usar un modelo más chico;
 * si está en `workMs`, es de inferencia y la respuesta es batchear o bajar
 * resolución. Sin separarlos, las dos se ven igual desde afuera: "la fase
 * tardó mucho".
 */
export interface PhaseTiming {
  /** Modelo que estuvo vivo durante la fase. */
  phase: string
  /** Descargar (si hizo falta), abrir el archivo y montar los pesos en RAM. */
  loadMs: number
  /** Inferencia propiamente dicha. */
  workMs: number
  /** Liberar la RAM antes de que entre el modelo siguiente. */
  unloadMs: number
}

/** Resultado completo de una corrida: los veredictos y dónde se fue el tiempo. */
export interface ReconciliationRun {
  verdicts: ReconciliationVerdict[]
  phases: PhaseTiming[]
  /** Reloj de pared de punta a punta. */
  elapsedMs: number
}

export interface ReconciliationVerdict {
  /** Nombre del archivo de factura auditado. */
  file: string
  status: PipelineStatus
  /** Datos extraídos; `null` si la extracción falló. */
  invoice: InvoiceData | null
  /** Veredicto ya verificado por el código; `null` si nunca se llegó a auditar. */
  audit: VerifiedAudit | null
  /** Documento de respaldo recuperado por RAG, si hubo alguno. */
  matchedSupportDoc: string | null
  /** Score de similitud de la búsqueda RAG. */
  supportScore: number | null
  /** Motivo del fallo cuando `status` es `ERROR`. */
  error: string | null
  timings: StageTiming[]
  totalMs: number
}

// ---------------------------------------------------------------------------
// JSON Schemas para structured outputs
// ---------------------------------------------------------------------------

/**
 * Convierte un schema zod a JSON Schema para `responseFormat`.
 *
 * llama.cpp compila este JSON Schema a una gramática GBNF que restringe la
 * generación token a token, así que se mantiene deliberadamente simple
 * (objetos, strings, números, enums y arrays) para no toparse con
 * construcciones que el conversor no soporte.
 */
function toJsonSchema(schema: z.ZodType): Record<string, unknown> {
  return z.toJSONSchema(schema, { target: 'draft-7' }) as Record<string, unknown>
}

export const INVOICE_JSON_SCHEMA = toJsonSchema(InvoiceDataSchema)
export const AUDIT_JSON_SCHEMA = toJsonSchema(AuditResultSchema)
