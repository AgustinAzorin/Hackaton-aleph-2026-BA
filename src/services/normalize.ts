/**
 * Capa de normalización de valores faltantes.
 *
 * Los schemas de extracción usan centinelas ("" para texto, 0 para números)
 * porque una gramática GBNF simple no admite claves opcionales ni nullables:
 * el modelo siempre emite todas las claves. Este módulo convierte esos
 * centinelas en `null` explícito ANTES de la verificación, para que el
 * verificador nunca confunda "no se pudo leer" con un valor real.
 *
 * Decisión documentada: un total de exactamente 0 se trata como evidencia
 * ilegible, no como una factura de monto cero. Una factura real por 0,00 es
 * un caso rarísimo, y el costo de confundir "ilegible" con "cero" es una
 * comparación numérica sin sentido — el sesgo conservador manda.
 */
import type { AuditResult, InvoiceData, LineItem } from '../types.js'

// ---------------------------------------------------------------------------
// Normalizadores de valor
// ---------------------------------------------------------------------------

/** "" (o sólo espacios) → null. Cualquier otro texto vuelve recortado. */
export function normalizeText(raw: string): string | null {
  const text = raw.trim()
  return text.length > 0 ? text : null
}

/**
 * Un monto centinela (0) o corrupto (negativo, NaN) → null. Ningún documento
 * de este dominio factura montos negativos: si aparece uno, el OCR falló.
 */
export function normalizeAmount(raw: number): number | null {
  return Number.isFinite(raw) && raw > 0 ? raw : null
}

/**
 * Normaliza un código de moneda a ISO de tres letras (mayúsculas); `null` si
 * el valor no parece un código legible. Acepta cualquier código ISO (ARS,
 * USD, EUR, ...): la comparación posterior es igualdad de strings
 * normalizados, sin lista blanca embebida en la lógica de comparación.
 */
export function normalizeCurrency(raw: string): string | null {
  const code = raw.trim().toUpperCase()
  return /^[A-Z]{3}$/.test(code) ? code : null
}

// ---------------------------------------------------------------------------
// Tipos internos: acá lo faltante es explícito
// ---------------------------------------------------------------------------

export interface NormalizedInvoice {
  invoiceNumber: string | null
  vendorName: string | null
  date: string | null
  totalAmount: number | null
  currency: string | null
  poReference: string | null
  items: LineItem[]
}

export interface NormalizedEvidence {
  supportDocumentId: string | null
  supportVendorName: string | null
  supportTotalAmount: number | null
  supportCurrency: string | null
  supportItems: LineItem[]
}

/** Convierte la salida cruda de extracción en el tipo interno con `null`. */
export function normalizeInvoice(raw: InvoiceData): NormalizedInvoice {
  return {
    invoiceNumber: normalizeText(raw.invoiceNumber),
    vendorName: normalizeText(raw.vendorName),
    date: normalizeText(raw.date),
    totalAmount: normalizeAmount(raw.totalAmount),
    currency: normalizeCurrency(raw.currency),
    poReference: normalizeText(raw.poReference),
    items: raw.items
  }
}

/** Convierte la transcripción cruda del respaldo en el tipo interno. */
export function normalizeEvidence(raw: AuditResult): NormalizedEvidence {
  return {
    supportDocumentId: normalizeText(raw.supportDocumentId),
    supportVendorName: normalizeText(raw.supportVendorName),
    supportTotalAmount: normalizeAmount(raw.supportTotalAmount),
    supportCurrency: normalizeCurrency(raw.supportCurrency),
    supportItems: raw.supportItems
  }
}

// ---------------------------------------------------------------------------
// Campos críticos
// ---------------------------------------------------------------------------

/**
 * Sin estos campos de la FACTURA no hay auditoría posible: no se sabe qué
 * documento es, de quién, por cuánto ni en qué moneda. Su ausencia degrada a
 * UNCERTAIN nombrando el campo. `date` y `poReference` NO son críticos: una
 * factura sin fecha legible o sin PO citado todavía puede compararse.
 */
export function missingCriticalFields(invoice: NormalizedInvoice): string[] {
  const missing: string[] = []
  if (invoice.invoiceNumber === null) missing.push('invoiceNumber')
  if (invoice.vendorName === null) missing.push('vendorName')
  if (invoice.totalAmount === null) missing.push('totalAmount')
  if (invoice.currency === null) missing.push('currency')
  return missing
}
