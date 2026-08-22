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
import { blocksToText, buildRetrievalQuery, listDocuments } from './services/qvacService.js'
import {
  AUDIT_JSON_SCHEMA,
  AuditResultSchema,
  INVOICE_JSON_SCHEMA,
  InvoiceDataSchema
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
console.log(`\n${passed} pruebas pasaron, ${failed} fallaron\n`)
process.exit(failed === 0 ? 0 : 1)
