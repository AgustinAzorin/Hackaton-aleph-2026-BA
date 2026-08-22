/**
 * Verificación offline del pipeline.
 *
 * Ejercita todo lo que no requiere descargar pesos de modelos: rasterización de
 * PDF, reconstrucción de texto desde bloques OCR, descubrimiento de documentos,
 * armado de la consulta RAG y los schemas de structured output. Sirve para
 * validar el pipeline en una máquina sin los modelos ya cacheados, y como
 * chequeo rápido antes de una demo.
 *
 * Uso: npm run verify
 */
import assert from 'node:assert/strict'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { toImagePages, isSupportedDocument } from './services/rasterize.js'
import {
  blocksToText,
  buildRetrievalQuery,
  listDocuments,
  sameVendor,
  supportLabel,
  verifyAgainstEvidence
} from './services/qvacService.js'
import {
  missingCriticalFields,
  normalizeAmount,
  normalizeCurrency,
  normalizeInvoice,
  normalizeText
} from './services/normalize.js'
import { canonicalPo, findCitedPo, resolvePoEvidence } from './services/retrieval.js'
import {
  AUDIT_JSON_SCHEMA,
  AuditResultSchema,
  INVOICE_JSON_SCHEMA,
  InvoiceDataSchema,
  type LineItem
} from './types.js'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const REPO_ROOT = path.resolve(__dirname, '..')

let passed = 0
let failed = 0

async function test(name: string, fn: () => void | Promise<void>): Promise<void> {
  try {
    await fn()
    passed++
    console.log(`  ✔ ${name}`)
  } catch (error) {
    failed++
    console.log(`  ✖ ${name}`)
    console.log(`     ${error instanceof Error ? error.message : String(error)}`)
  }
}

console.log('\nVerificación offline del reconciliador\n')

// ---------------------------------------------------------------------------
console.log('Descubrimiento de documentos')

await test('acepta PDF, PNG y JPG; rechaza el resto', () => {
  assert.equal(isSupportedDocument('factura.pdf'), true)
  assert.equal(isSupportedDocument('factura.PNG'), true)
  assert.equal(isSupportedDocument('factura.jpeg'), true)
  assert.equal(isSupportedDocument('notas.txt'), false)
  assert.equal(isSupportedDocument('hoja.xlsx'), false)
})

await test('lista las 6 facturas y las 5 órdenes de compra de samples/', async () => {
  const invoices = await listDocuments(path.join(REPO_ROOT, 'samples/invoices'))
  const support = await listDocuments(path.join(REPO_ROOT, 'samples/support'))
  assert.equal(invoices.length, 6, `esperaba 6 facturas, encontré ${invoices.length}`)
  assert.equal(support.length, 5, `esperaba 5 órdenes de compra, encontré ${support.length}`)
  // Mezcla real de formatos: la corrida debe ejercitar ambas rutas de entrada.
  assert.ok(invoices.some((f) => f.endsWith('.pdf')), 'no hay ninguna factura en PDF')
  assert.ok(invoices.some((f) => f.endsWith('.png')), 'no hay ninguna factura en PNG')
})

// ---------------------------------------------------------------------------
console.log('\nRasterización de PDF')

await test('un PDF de una página produce exactamente un PNG', async () => {
  const pages = await toImagePages(path.join(REPO_ROOT, 'samples/invoices/INV-1001.pdf'))
  assert.equal(pages.length, 1)
  assert.ok(pages[0]?.endsWith('.png'), `esperaba un PNG, obtuve ${pages[0]}`)
})

await test('una imagen se deja pasar sin rasterizar', async () => {
  const original = path.join(REPO_ROOT, 'samples/invoices/INV-1002.png')
  const pages = await toImagePages(original)
  assert.deepEqual(pages, [original])
})

// ---------------------------------------------------------------------------
console.log('\nReconstrucción de texto desde bloques OCR')

await test('reordena una fila de tabla desordenada en orden de lectura', () => {
  // El motor OCR puede devolver los bloques en cualquier orden; lo que importa
  // es que la descripción quede pegada a su cantidad y a su monto.
  const text = blocksToText([
    { text: '680.00', bbox: [470, 100, 520, 114] },
    { text: 'Circular saw blade 190mm', bbox: [50, 101, 300, 115] },
    { text: '34.00', bbox: [380, 100, 420, 114] },
    { text: '20', bbox: [320, 100, 340, 114] }
  ])
  assert.equal(text, 'Circular saw blade 190mm  20  34.00  680.00')
})

await test('separa filas distintas en líneas distintas', () => {
  const text = blocksToText([
    { text: 'Safety goggles', bbox: [50, 140, 200, 154] },
    { text: 'Circular saw blade', bbox: [50, 100, 200, 114] },
    { text: '660.00', bbox: [470, 140, 520, 154] },
    { text: '680.00', bbox: [470, 100, 520, 114] }
  ])
  assert.equal(text, 'Circular saw blade  680.00\nSafety goggles  660.00')
})

await test('sin bounding boxes conserva el orden de llegada', () => {
  const text = blocksToText([{ text: 'INVOICE' }, { text: 'INV-1001' }])
  assert.equal(text, 'INVOICE\nINV-1001')
})

await test('descarta bloques vacíos sin dejar líneas en blanco', () => {
  const text = blocksToText([
    { text: 'TOTAL', bbox: [380, 200, 430, 214] },
    { text: '   ', bbox: [440, 200, 450, 214] },
    { text: 'USD 2,980.00', bbox: [470, 200, 560, 214] }
  ])
  assert.equal(text, 'TOTAL  USD 2,980.00')
})

// ---------------------------------------------------------------------------
console.log('\nConsulta de recuperación RAG')

await test('se queda con el encabezado, donde viven proveedor y referencia de PO', () => {
  const invoiceText = [
    'INVOICE',
    'Cedar Hardware Supply',
    'Invoice Number: INV-1004',
    'PO Reference: PO-5004',
    ...Array.from({ length: 40 }, (_, i) => `Línea de ítem irrelevante ${i}`)
  ].join('\n')

  const query = buildRetrievalQuery(invoiceText)
  assert.ok(query.includes('PO-5004'), 'la consulta perdió la referencia de PO')
  assert.ok(query.includes('Cedar Hardware Supply'), 'la consulta perdió el proveedor')
  assert.ok(query.length <= 800, 'la consulta excede el tope de 800 caracteres')
  assert.ok(!query.includes('irrelevante 30'), 'la consulta arrastra ruido del cuerpo de la tabla')
})

// ---------------------------------------------------------------------------
console.log('\nStructured outputs')

await test('el JSON Schema de factura es apto para gramática GBNF', () => {
  const schema = INVOICE_JSON_SCHEMA as {
    type: string
    properties: Record<string, unknown>
    required: string[]
  }
  assert.equal(schema.type, 'object')
  // Todos los campos deben ser obligatorios: un campo opcional le permite al
  // modelo omitir la clave y esquivar la restricción.
  for (const field of [
    'invoiceNumber',
    'vendorName',
    'date',
    'totalAmount',
    'currency',
    'poReference',
    'items'
  ]) {
    assert.ok(field in schema.properties, `falta la propiedad ${field}`)
    assert.ok(schema.required.includes(field), `${field} debería ser obligatorio`)
  }
})

await test('el JSON Schema de auditoría fija el veredicto a los tres valores válidos', () => {
  const schema = AUDIT_JSON_SCHEMA as {
    properties: { verdict: { enum?: string[] } }
    required: string[]
  }
  assert.deepEqual(schema.properties.verdict.enum, ['MATCH', 'DISCREPANCY', 'UNCERTAIN'])
  for (const field of ['verdict', 'confidence', 'summary', 'discrepancies']) {
    assert.ok(schema.required.includes(field), `${field} debería ser obligatorio`)
  }
})

await test('zod acepta una factura bien formada', () => {
  const result = InvoiceDataSchema.safeParse({
    invoiceNumber: 'INV-1004',
    vendorName: 'Cedar Hardware Supply',
    date: '2026-07-24',
    totalAmount: 2980,
    currency: 'USD',
    poReference: 'PO-5004',
    items: [{ description: 'Circular saw blade 190mm', quantity: 20, unitPrice: 34, amount: 680 }]
  })
  assert.ok(result.success, 'una factura válida fue rechazada')
})

await test('zod rechaza un total como string y explica por qué', () => {
  // Este es el caso que dispara el reintento: la respuesta debe llegar con el
  // motivo concreto para poder devolvérselo al modelo.
  const result = InvoiceDataSchema.safeParse({
    invoiceNumber: 'INV-1004',
    vendorName: 'Cedar Hardware Supply',
    date: '2026-07-24',
    totalAmount: 'USD 2,980.00',
    currency: 'USD',
    poReference: 'PO-5004',
    items: []
  })
  assert.equal(result.success, false)
  const issue = result.error?.issues.find((i) => i.path[0] === 'totalAmount')
  assert.ok(issue, 'el error no señala el campo totalAmount')
})

await test('zod rechaza un veredicto fuera del enum', () => {
  const result = AuditResultSchema.safeParse({
    verdict: 'PROBABLY_FINE',
    confidence: 0.9,
    summary: 'Parece correcto.',
    discrepancies: []
  })
  assert.equal(result.success, false)
})

// ---------------------------------------------------------------------------
console.log('\nVerificación determinista (los seis casos de samples/)')

const item = (description: string, quantity: number, unitPrice: number) => ({
  description,
  quantity,
  unitPrice,
  amount: quantity * unitPrice
})

/** Factura con ítems que suman su total, salvo que se indique lo contrario. */
const invoiceOf = (
  vendorName: string,
  poReference: string,
  totalAmount: number,
  items: LineItem[],
  currency = 'USD'
) => ({
  invoiceNumber: 'INV-0000',
  vendorName,
  date: '2026-07-01',
  totalAmount,
  currency,
  poReference,
  items
})

/** Transcripción del respaldo tal como la devuelve el modelo. */
const supportOf = (
  supportDocumentId: string,
  supportVendorName: string,
  supportTotalAmount: number,
  supportItems: LineItem[],
  supportCurrency = 'USD'
) => ({
  supportDocumentId,
  supportVendorName,
  supportTotalAmount,
  supportCurrency,
  supportItems,
  discrepancies: [],
  verdict: 'MATCH' as const,
  confidence: 0.9,
  summary: 'Todo coincide.'
})

await test('INV-1001: coincidencia real se mantiene MATCH', () => {
  const items = [
    item('A4 Copy Paper, 80gsm, ream', 40, 6.5),
    item('Toner Cartridge HP 26X', 8, 145),
    item('Stapler, heavy duty', 10, 42)
  ]
  const result = verifyAgainstEvidence(
    invoiceOf('Acme Office Supplies LLC', 'PO-5001', 1840, items),
    supportOf('PO-5001.pdf', 'Acme Office Supplies LLC', 1840, items),
    0.876
  )
  assert.equal(result.verdict, 'MATCH')
  assert.deepEqual(result.discrepancies, [])
})

await test('INV-1002: coincidencia exacta con decimales se mantiene MATCH', () => {
  const items = [
    item('Hex Bolt M12 x 60mm, box of 100', 25, 78),
    item('Lock Washer M12, box of 500', 15, 22.7),
    item('Threadlocker adhesive 50ml', 12, 82.5)
  ]
  const result = verifyAgainstEvidence(
    invoiceOf('Bolt & Nut Co.', 'PO-5002', 3280.5, items),
    supportOf('PO-5002.pdf', 'Bolt & Nut Co.', 3280.5, items),
    0.886
  )
  assert.equal(result.verdict, 'MATCH')
  assert.deepEqual(result.discrepancies, [])
})

await test('un renglón mal leído por el OCR no dispara una acusación falsa', () => {
  // Caso real de la tercera corrida: la extracción leyó mal un importe y el
  // chequeo de coherencia interna reportó un desvío de 1.500 que no existía.
  // Si el importe de un renglón no cierra con cantidad × precio, el que no es
  // confiable es el OCR, no la factura.
  const invoiceItems = [
    item('Container drayage, port to warehouse', 6, 520),
    { description: 'Palletizing service', quantity: 12, unitPrice: 90, amount: 2580 },
    item('Fuel surcharge', 1, 420)
  ]
  const result = verifyAgainstEvidence(
    invoiceOf('Northwind Logistics Inc.', 'PO-5003', 4620, invoiceItems),
    supportOf('PO-5003.pdf', 'Northwind Logistics Inc.', 4620, [
      item('Container drayage, port to warehouse', 6, 520),
      item('Palletizing service', 12, 90),
      item('Fuel surcharge', 1, 420)
    ]),
    0.859
  )
  assert.ok(
    !result.discrepancies.some((d) => d.field === 'coherencia interna'),
    'acusó a la factura por un renglón que el OCR leyó mal'
  )
})

await test('INV-1003: recargo no autorizado da DISCREPANCY', () => {
  const supportItems = [
    item('Container drayage, port to warehouse', 6, 520),
    item('Palletizing service', 12, 90)
  ]
  const invoiceItems = [...supportItems, item('Fuel surcharge', 1, 420)]
  const result = verifyAgainstEvidence(
    invoiceOf('Northwind Logistics Inc.', 'PO-5003', 4620, invoiceItems),
    supportOf('PO-5003.pdf', 'Northwind Logistics Inc.', 4200, supportItems),
    0.859
  )
  assert.equal(result.verdict, 'DISCREPANCY')
  assert.ok(
    result.discrepancies.some((d) => d.field.includes('Fuel surcharge')),
    'no señaló el ítem no autorizado'
  )
  assert.ok(result.discrepancies.some((d) => d.field === 'total'), 'no señaló el total')
})

await test('INV-1004: totales iguales, pero falta un ítem del respaldo', () => {
  // El caso que el modelo no detectó dos corridas seguidas. Los totales
  // coinciden en 2920, así que sólo lo delatan el cruce de ítems y el hecho
  // de que los ítems facturados sumen 2100 contra un total de 2920.
  const invoiceItems = [
    item('Circular saw blade 190mm', 20, 34),
    item('Safety goggles, polycarbonate', 60, 11),
    item('Work gloves, leather, pair', 80, 9.5)
  ]
  const supportItems = [...invoiceItems, item('Cordless drill 18V', 4, 205)]
  const result = verifyAgainstEvidence(
    invoiceOf('Cedar Hardware Supply', 'PO-5004', 2920, invoiceItems),
    supportOf('PO-5004.pdf', 'Cedar Hardware Supply', 2920, supportItems),
    0.885
  )
  assert.equal(result.verdict, 'DISCREPANCY')
  assert.ok(
    result.discrepancies.some((d) => d.field.includes('Cordless drill 18V')),
    'no señaló el ítem no facturado'
  )
  assert.ok(
    result.discrepancies.some((d) => d.field === 'coherencia interna'),
    'no señaló que los ítems no suman el total'
  )
})

await test('INV-1005: respaldo de otro proveedor da UNCERTAIN, no DISCREPANCY', () => {
  // La regresión más grave de la corrida anterior: se acusó a Quantum Freight
  // comparándola contra una orden de compra de Northwind.
  const result = verifyAgainstEvidence(
    invoiceOf('Quantum Freight Systems', '', 3835, [
      item('Expedited air freight, 3 pallets', 3, 1150),
      item('Customs brokerage fee', 1, 385)
    ]),
    supportOf('PO-5003.pdf', 'Northwind Logistics Inc.', 4200, [
      item('Container drayage, port to warehouse', 6, 520)
    ]),
    0.692
  )
  assert.equal(result.verdict, 'UNCERTAIN')
  assert.deepEqual(result.discrepancies, [], 'no debe acusar con evidencia ajena')
  assert.ok(
    result.summary.includes('Northwind'),
    `el resumen debe explicar el cruce erróneo: ${result.summary}`
  )
})

await test('INV-1006: precio unitario inflado da DISCREPANCY', () => {
  const result = verifyAgainstEvidence(
    invoiceOf('Acme Office Supplies LLC', 'PO-5006', 915, [
      item('Archive box, corrugated', 50, 13.5),
      item('Label sheets, 100 per pack', 30, 8)
    ]),
    supportOf('PO-5006.pdf', 'Acme Office Supplies LLC', 840, [
      item('Archive box, corrugated', 50, 12),
      item('Label sheets, 100 per pack', 30, 8)
    ]),
    0.873
  )
  assert.equal(result.verdict, 'DISCREPANCY')
  assert.ok(
    result.discrepancies.some((d) => d.field.startsWith('precio unitario')),
    'no señaló el precio unitario'
  )
})

// ---------------------------------------------------------------------------
console.log('\nSalvaguardas de evidencia insuficiente')

await test('sin respaldo identificado el veredicto es UNCERTAIN', () => {
  const result = verifyAgainstEvidence(
    invoiceOf('Quantum Freight Systems', '', 3835, []),
    supportOf('', '', 0, []),
    null
  )
  assert.equal(result.verdict, 'UNCERTAIN')
})

await test('un total de respaldo ilegible impide declarar MATCH', () => {
  const result = verifyAgainstEvidence(
    invoiceOf('Acme Office Supplies LLC', 'PO-5001', 1840, []),
    supportOf('PO-5001.pdf', 'Acme Office Supplies LLC', 0, []),
    0.88
  )
  assert.equal(result.verdict, 'UNCERTAIN')
})

await test('una referencia de PO que no coincide con el respaldo da UNCERTAIN', () => {
  const result = verifyAgainstEvidence(
    invoiceOf('Acme Office Supplies LLC', 'PO-5001', 1840, []),
    supportOf('PO-5006.pdf', 'Acme Office Supplies LLC', 840, []),
    0.8
  )
  assert.equal(result.verdict, 'UNCERTAIN')
  assert.ok(result.summary.includes('PO-5001'))
})

await test('proveedor sin confirmar y recuperación floja da UNCERTAIN', () => {
  const result = verifyAgainstEvidence(
    invoiceOf('Quantum Freight Systems', '', 3835, []),
    supportOf('PO-5003.pdf', '', 4200, []),
    0.69
  )
  assert.equal(result.verdict, 'UNCERTAIN')
})

await test('diferencias de centavo se toleran como redondeo', () => {
  const items = [item('A4 Copy Paper', 40, 6.5)]
  const result = verifyAgainstEvidence(
    invoiceOf('Acme Office Supplies LLC', 'PO-5001', 260.004, items),
    supportOf('PO-5001.pdf', 'Acme Office Supplies LLC', 260, items),
    0.88
  )
  assert.equal(result.verdict, 'MATCH')
})

await test('el sufijo societario no impide reconocer al mismo proveedor', () => {
  const items = [item('Palletizing service', 12, 90)]
  const result = verifyAgainstEvidence(
    invoiceOf('Northwind Logistics Inc.', 'PO-5003', 1080, items),
    supportOf('PO-5003.pdf', 'Northwind Logistics', 1080, items),
    0.86
  )
  assert.equal(result.verdict, 'MATCH')
})

await test('una variación menor de OCR no convierte un ítem en faltante', () => {
  const result = verifyAgainstEvidence(
    invoiceOf('Cedar Hardware Supply', 'PO-5004', 680, [
      item('Circular saw blade 190mm', 20, 34)
    ]),
    supportOf('PO-5004.pdf', 'Cedar Hardware Supply', 680, [
      item('Circular saw blade, 190 mm', 20, 34)
    ]),
    0.88
  )
  assert.equal(result.verdict, 'MATCH', `descripciones equivalentes se trataron como distintas`)
})

// ---------------------------------------------------------------------------
console.log('\nNormalización de valores faltantes (centinelas "" y 0 → null)')

await test('los centinelas de texto y monto se convierten en null explícito', () => {
  assert.equal(normalizeText(''), null)
  assert.equal(normalizeText('   '), null)
  assert.equal(normalizeText(' PO-5001 '), 'PO-5001')
  // Un total de exactamente 0 es evidencia ilegible, no una factura de 0,00:
  // decisión conservadora documentada en normalize.ts.
  assert.equal(normalizeAmount(0), null)
  assert.equal(normalizeAmount(-5), null)
  assert.equal(normalizeAmount(Number.NaN), null)
  assert.equal(normalizeAmount(1840), 1840)
})

await test('la moneda se normaliza a ISO de tres letras o null', () => {
  assert.equal(normalizeCurrency(' usd '), 'USD')
  assert.equal(normalizeCurrency('ARS'), 'ARS')
  assert.equal(normalizeCurrency('eur'), 'EUR')
  assert.equal(normalizeCurrency(''), null)
  assert.equal(normalizeCurrency('$'), null)
  assert.equal(normalizeCurrency('dolares'), null)
})

await test('missingCriticalFields nombra exactamente lo que falta', () => {
  const complete = normalizeInvoice(invoiceOf('Acme', 'PO-5001', 100, []))
  assert.deepEqual(missingCriticalFields(complete), [])

  const broken = normalizeInvoice({
    invoiceNumber: '',
    vendorName: 'Acme',
    date: '',
    totalAmount: 0,
    currency: 'USD',
    poReference: '',
    items: []
  })
  assert.deepEqual(missingCriticalFields(broken), ['invoiceNumber', 'totalAmount'])
})

await test('un total de factura ilegible (0) da UNCERTAIN nombrando el campo', () => {
  const result = verifyAgainstEvidence(
    invoiceOf('Acme Office Supplies LLC', 'PO-5001', 0, []),
    supportOf('PO-5001.pdf', 'Acme Office Supplies LLC', 1840, []),
    0.88
  )
  assert.equal(result.verdict, 'UNCERTAIN')
  assert.deepEqual(result.discrepancies, [])
  assert.ok(result.summary.includes('totalAmount'), `debe nombrar el campo faltante: ${result.summary}`)
})

await test('un proveedor de factura ilegible da UNCERTAIN, no una comparación a ciegas', () => {
  const result = verifyAgainstEvidence(
    invoiceOf('', 'PO-5001', 1840, []),
    supportOf('PO-5001.pdf', 'Acme Office Supplies LLC', 1840, []),
    0.88
  )
  assert.equal(result.verdict, 'UNCERTAIN')
  assert.ok(result.summary.includes('vendorName'))
})

await test('una moneda de factura ilegible da UNCERTAIN nombrando el campo', () => {
  const result = verifyAgainstEvidence(
    invoiceOf('Acme Office Supplies LLC', 'PO-5001', 1840, [], ''),
    supportOf('PO-5001.pdf', 'Acme Office Supplies LLC', 1840, []),
    0.88
  )
  assert.equal(result.verdict, 'UNCERTAIN')
  assert.ok(result.summary.includes('currency'), `debe nombrar la moneda faltante: ${result.summary}`)
})

await test('fecha y PO faltantes NO bloquean por sí solos el veredicto', () => {
  // Campos no críticos: la factura sin fecha ni PO citado todavía se compara.
  const items = [item('A4 Copy Paper, 80gsm, ream', 40, 6.5)]
  const invoice = { ...invoiceOf('Acme Office Supplies LLC', '', 260, items), date: '' }
  const result = verifyAgainstEvidence(
    invoice,
    supportOf('PO-5001.pdf', 'Acme Office Supplies LLC', 260, items),
    0.88
  )
  assert.equal(result.verdict, 'MATCH')
})

await test('una factura sin ítems detallados aún permite comparar totales', () => {
  const result = verifyAgainstEvidence(
    invoiceOf('Acme Office Supplies LLC', 'PO-5001', 2000, []),
    supportOf('PO-5001.pdf', 'Acme Office Supplies LLC', 1840, []),
    0.88
  )
  assert.equal(result.verdict, 'DISCREPANCY')
  assert.ok(result.discrepancies.some((d) => d.field === 'total'))
})

// ---------------------------------------------------------------------------
console.log('\nMonedas (los montos sólo son comparables en la misma moneda)')

await test('misma moneda en ambos documentos permite MATCH', () => {
  const items = [item('A4 Copy Paper, 80gsm, ream', 40, 6.5)]
  const result = verifyAgainstEvidence(
    invoiceOf('Acme Office Supplies LLC', 'PO-5001', 260, items, 'USD'),
    supportOf('PO-5001.pdf', 'Acme Office Supplies LLC', 260, items, 'usd '),
    0.88
  )
  assert.equal(result.verdict, 'MATCH', 'la normalización de moneda no debería distinguir mayúsculas')
})

await test('monedas distintas con montos iguales NO es MATCH: es UNCERTAIN sin acusaciones', () => {
  // ARS 1000 contra USD 1000 no son comparables; convertir implícitamente
  // sería inventar un tipo de cambio que el sistema no tiene.
  const items = [item('Servicio de flete', 1, 1000)]
  const result = verifyAgainstEvidence(
    invoiceOf('Northwind Logistics Inc.', 'PO-5003', 1000, items, 'ARS'),
    supportOf('PO-5003.pdf', 'Northwind Logistics Inc.', 1000, items, 'USD'),
    0.88
  )
  assert.equal(result.verdict, 'UNCERTAIN')
  assert.deepEqual(result.discrepancies, [], 'una moneda distinta no debe producir acusaciones numéricas')
  assert.ok(result.summary.includes('ARS') && result.summary.includes('USD'), `el resumen debe explicar las monedas: ${result.summary}`)
})

await test('monedas distintas también bloquean cuando los montos difieren', () => {
  const result = verifyAgainstEvidence(
    invoiceOf('Northwind Logistics Inc.', 'PO-5003', 999999, [], 'EUR'),
    supportOf('PO-5003.pdf', 'Northwind Logistics Inc.', 1000, [], 'ARS'),
    0.88
  )
  assert.equal(result.verdict, 'UNCERTAIN')
  assert.deepEqual(result.discrepancies, [])
})

await test('moneda del respaldo ilegible: los montos se comparan pero el resumen lo advierte', () => {
  const items = [item('A4 Copy Paper, 80gsm, ream', 40, 6.5)]
  const result = verifyAgainstEvidence(
    invoiceOf('Acme Office Supplies LLC', 'PO-5001', 260, items, 'USD'),
    supportOf('PO-5001.pdf', 'Acme Office Supplies LLC', 260, items, ''),
    0.88
  )
  assert.equal(result.verdict, 'MATCH', 'sin moneda de respaldo los chequeos numéricos deben seguir corriendo')
  assert.ok(result.summary.includes('sin verificar'), `el resumen debe advertir la moneda sin verificar: ${result.summary}`)
})

await test('moneda del respaldo ilegible no tapa una discrepancia numérica', () => {
  const result = verifyAgainstEvidence(
    invoiceOf('Acme Office Supplies LLC', 'PO-5006', 915, []),
    supportOf('PO-5006.pdf', 'Acme Office Supplies LLC', 840, [], ''),
    0.87
  )
  assert.equal(result.verdict, 'DISCREPANCY')
  assert.ok(result.summary.includes('sin verificar'))
})

// ---------------------------------------------------------------------------
console.log('\nIdentidad de proveedor (jerarquía conservadora)')

await test('la normalización iguala variantes de puntuación y sufijo societario', () => {
  assert.equal(sameVendor('ACME S.A.', 'acme sa'), true)
  assert.equal(sameVendor('ACME S.A.', 'ACME S.A'), true)
  assert.equal(sameVendor('acme sa', 'ACME S.A'), true)
})

await test('compartir un solo rubro NO prueba identidad', () => {
  // La regresión que motiva el cambio: con la regla vieja de "cualquier token
  // compartido", estas dos empresas distintas se consideraban la misma.
  assert.equal(sameVendor('Acme Logistics', 'Beta Logistics'), false)
  assert.equal(sameVendor('Quantum Freight Systems', 'Northwind Logistics Inc.'), false)
})

await test('el sufijo societario no impide reconocer al mismo proveedor (nivel 1)', () => {
  assert.equal(sameVendor('Northwind Logistics Inc.', 'Northwind Logistics'), true)
})

await test('un token garbleado por OCR en un nombre de tres palabras aún coincide', () => {
  assert.equal(sameVendor('Quantum Freight Syst3ms', 'Quantum Freight Systems'), true)
})

await test('en un nombre de dos palabras, un token garbleado degrada a distinto (conservador)', () => {
  // Con la mitad del nombre ilegible ya no hay mayoría de tokens compartidos.
  // El costo es un UNCERTAIN de más — nunca una acusación contra el proveedor
  // equivocado, que es el fallo caro.
  assert.equal(sameVendor('Northwind Log1stics', 'Northwind Logistics'), false)
})

await test('un proveedor distinto degrada a UNCERTAIN, jamás a DISCREPANCY', () => {
  const result = verifyAgainstEvidence(
    invoiceOf('Acme Logistics', 'PO-5003', 4620, []),
    supportOf('PO-5003.pdf', 'Beta Logistics', 4200, []),
    0.88
  )
  assert.equal(result.verdict, 'UNCERTAIN')
  assert.deepEqual(result.discrepancies, [])
})

// ---------------------------------------------------------------------------
console.log('\nRecuperación híbrida (PO exacto primero, semántica como último recurso)')

await test('la forma canónica de un PO ignora prefijo, guiones y mayúsculas', () => {
  assert.equal(canonicalPo('PO-5003'), '5003')
  assert.equal(canonicalPo('po 5003'), '5003')
  assert.equal(canonicalPo('PO5003'), '5003')
  assert.equal(canonicalPo('5003'), '5003')
  assert.equal(canonicalPo('PO-AB-123'), 'AB123')
})

await test('encuentra el PO citado en el texto OCR de una factura', () => {
  const cited = findCitedPo('INVOICE\nCedar Hardware Supply\nInvoice Number: INV-1004\nPO Reference: PO-5004\nDate: 2026-07-24')
  assert.ok(cited !== null, 'no encontró la cita')
  assert.equal(cited.canonical, '5004')
})

await test('"P.O. Box" de una dirección no se confunde con una cita de PO', () => {
  assert.equal(findCitedPo('Acme LLC\nP.O. Box Newark NJ\nInvoice INV-9'), null)
  assert.equal(findCitedPo('factura sin ninguna referencia'), null)
})

const supportSet = [
  { file: '/support/PO-5001.pdf', text: 'PURCHASE ORDER\nAcme Office Supplies LLC\nPO Number: PO-5001\nTOTAL USD 1,840.00' },
  { file: '/support/PO-5003.pdf', text: 'PURCHASE ORDER\nNorthwind Logistics Inc.\nPO Number: PO-5003\nTOTAL USD 4,200.00' },
  { file: '/support/orden_julio.pdf', text: 'PURCHASE ORDER\nBolt & Nut Co.\nPO Number: PO-7788\nTOTAL USD 900.00' }
]

await test('un PO citado se resuelve exacto contra el nombre de archivo', () => {
  const result = resolvePoEvidence('INVOICE\nAcme\nPO Reference: PO-5001\nTOTAL 1840', supportSet)
  assert.equal(result.kind, 'exact')
  assert.ok(result.kind === 'exact' && result.doc.file.endsWith('PO-5001.pdf'))
})

await test('un PO citado se resuelve exacto contra el texto OCR aunque el archivo tenga otro nombre', () => {
  const result = resolvePoEvidence('INVOICE\nBolt & Nut Co.\nPO Reference: PO-7788', supportSet)
  assert.equal(result.kind, 'exact')
  assert.ok(result.kind === 'exact' && result.doc.file.endsWith('orden_julio.pdf'))
})

await test('un PO citado ausente del conjunto ENTERO da not-found, nunca "algo parecido"', () => {
  // El conjunto contiene órdenes muy similares en contenido; nada de eso
  // importa: la cita es un identificador y el identificador no está.
  const result = resolvePoEvidence('INVOICE\nNorthwind Logistics Inc.\nPO Reference: PO-5099\nContainer drayage, port to warehouse', supportSet)
  assert.equal(result.kind, 'not-found')
  assert.ok(result.kind === 'not-found' && result.citedPo.includes('5099'))
})

await test('sin PO citado la resolución delega en la búsqueda semántica', () => {
  const result = resolvePoEvidence('INVOICE\nQuantum Freight Systems\nExpedited air freight', supportSet)
  assert.equal(result.kind, 'no-po')
})

await test('con dos respaldos de contenido casi idéntico gana el del PO citado', () => {
  const twins = [
    { file: '/support/PO-9001.pdf', text: 'PURCHASE ORDER\nAcme LLC\nPO Number: PO-9001\nWidget A 10 x 5.00\nTOTAL USD 50.00' },
    { file: '/support/PO-9002.pdf', text: 'PURCHASE ORDER\nAcme LLC\nPO Number: PO-9002\nWidget A 10 x 5.00\nTOTAL USD 50.00' }
  ]
  const result = resolvePoEvidence('INVOICE\nAcme LLC\nPO Reference: PO-9002', twins)
  assert.equal(result.kind, 'exact')
  assert.ok(result.kind === 'exact' && result.doc.file.endsWith('PO-9002.pdf'))
})

await test('el verificador rechaza un respaldo de PO equivocado aunque la similitud sea alta', () => {
  // Cinturón y tiradores: si a pesar de todo llegara un respaldo ajeno con
  // score semántico alto, la referencia citada sigue mandando.
  const result = verifyAgainstEvidence(
    invoiceOf('Northwind Logistics Inc.', 'PO-5099', 4620, []),
    supportOf('PO-5003.pdf', 'Northwind Logistics Inc.', 4200, []),
    0.95
  )
  assert.equal(result.verdict, 'UNCERTAIN')
  assert.deepEqual(result.discrepancies, [])
})

// ---------------------------------------------------------------------------
console.log('\nEtiqueta del documento de respaldo')

await test('recupera el nombre de archivo del prefijo del fragmento', () => {
  assert.equal(supportLabel('[PO-5003.pdf]\nPURCHASE ORDER\nTOTAL USD 4,200.00'), 'PO-5003.pdf')
  assert.equal(supportLabel('sin prefijo alguno'), null)
})

// ---------------------------------------------------------------------------
console.log('\nOrden del schema de auditoría')

await test('la evidencia precede al veredicto en el JSON Schema', () => {
  // Es el arreglo central: la gramática GBNF emite las claves en el orden del
  // schema, así que el veredicto no puede salir antes que la evidencia.
  const schema = AUDIT_JSON_SCHEMA as { properties: Record<string, unknown> }
  const keys = Object.keys(schema.properties)
  const verdictAt = keys.indexOf('verdict')
  for (const evidence of [
    'supportDocumentId',
    'supportVendorName',
    'supportTotalAmount',
    'supportCurrency',
    'supportItems',
    'discrepancies'
  ]) {
    assert.ok(
      keys.indexOf(evidence) < verdictAt,
      `${evidence} debe emitirse antes que verdict, pero sale después`
    )
  }
})

// ---------------------------------------------------------------------------
console.log(`\n${passed} pruebas pasaron, ${failed} fallaron\n`)
process.exit(failed === 0 ? 0 : 1)
