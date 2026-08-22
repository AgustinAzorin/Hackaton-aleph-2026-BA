/**
 * M0 — Generador de datos de prueba.
 *
 * Produce facturas y órdenes de compra sintéticas con discrepancias plantadas
 * deliberadamente, para tener ground truth verificable del pipeline.
 *
 *   samples/invoices/  → facturas a auditar (PDF y PNG)
 *   samples/support/   → órdenes de compra de respaldo (PDF)
 *
 * Uso: npm run samples
 */
import PDFDocument from 'pdfkit'
import { createWriteStream } from 'node:fs'
import { mkdir, rm, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const INVOICES_DIR = path.join(__dirname, 'invoices')
const SUPPORT_DIR = path.join(__dirname, 'support')

interface Item {
  description: string
  quantity: number
  unitPrice: number
}

interface Doc {
  kind: 'INVOICE' | 'PURCHASE ORDER'
  /** Número de factura u orden de compra. */
  number: string
  vendor: string
  vendorAddress: string
  date: string
  /** Referencia cruzada: la factura apunta a su PO. */
  reference?: string
  items: Item[]
  /**
   * Total impreso en el documento. Se declara explícitamente (en vez de
   * calcularse) para poder plantar un total que NO coincide con la suma de
   * ítems — así se prueba que el auditor detecte el desvío.
   */
  printedTotal: number
  currency: string
  notes?: string
}

const money = (n: number, currency: string) =>
  `${currency} ${n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`

/** Renderiza un documento financiero a PDF, con tipografía legible por OCR. */
function renderPdf(doc: Doc, outPath: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const pdf = new PDFDocument({ size: 'A4', margin: 50 })
    const stream = createWriteStream(outPath)
    pdf.pipe(stream)
    stream.on('finish', () => resolve())
    stream.on('error', reject)

    // Encabezado
    pdf.font('Helvetica-Bold').fontSize(22).text(doc.kind, { align: 'left' })
    pdf.moveDown(0.3)
    pdf.font('Helvetica-Bold').fontSize(13).text(doc.vendor)
    pdf.font('Helvetica').fontSize(10).text(doc.vendorAddress)
    pdf.moveDown(1)

    // Metadatos
    const label = doc.kind === 'INVOICE' ? 'Invoice Number' : 'PO Number'
    pdf.font('Helvetica').fontSize(11)
    pdf.text(`${label}: ${doc.number}`)
    pdf.text(`Date: ${doc.date}`)
    if (doc.reference) pdf.text(`PO Reference: ${doc.reference}`)
    pdf.text(`Currency: ${doc.currency}`)
    pdf.moveDown(1)

    // Tabla de ítems
    const top = pdf.y
    const cols = { desc: 50, qty: 320, unit: 380, amount: 470 }
    pdf.font('Helvetica-Bold').fontSize(11)
    pdf.text('Description', cols.desc, top)
    pdf.text('Qty', cols.qty, top)
    pdf.text('Unit Price', cols.unit, top)
    pdf.text('Amount', cols.amount, top)
    pdf
      .moveTo(50, top + 16)
      .lineTo(545, top + 16)
      .stroke()

    let y = top + 26
    pdf.font('Helvetica').fontSize(10)
    for (const item of doc.items) {
      const amount = item.quantity * item.unitPrice
      pdf.text(item.description, cols.desc, y, { width: 260 })
      pdf.text(String(item.quantity), cols.qty, y)
      pdf.text(item.unitPrice.toFixed(2), cols.unit, y)
      pdf.text(amount.toFixed(2), cols.amount, y)
      y += 22
    }

    pdf
      .moveTo(50, y + 4)
      .lineTo(545, y + 4)
      .stroke()
    y += 16

    // Total impreso
    pdf.font('Helvetica-Bold').fontSize(12)
    pdf.text('TOTAL', cols.unit, y)
    pdf.text(money(doc.printedTotal, doc.currency), cols.amount - 30, y, { width: 145 })

    if (doc.notes) {
      pdf.moveDown(3)
      pdf.font('Helvetica').fontSize(9).text(doc.notes, 50, y + 50, { width: 495 })
    }

    pdf.end()
  })
}

/**
 * Rasteriza un PDF a PNG. Se usa para entregar algunas facturas como imagen
 * escaneada, de modo que el pipeline ejercite tanto la rama PDF como la PNG.
 */
async function rasterizeToPng(pdfPath: string, pngPath: string): Promise<void> {
  const { pdf } = await import('pdf-to-img')
  const document = await pdf(pdfPath, { scale: 2 })
  for await (const page of document) {
    await writeFile(pngPath, page)
    break // documentos de una sola página
  }
}

// ---------------------------------------------------------------------------
// Casos de prueba: 2 matches, 3 discrepancias, 1 factura sin PO.
// ---------------------------------------------------------------------------

const ACME_ADDR = 'Acme Office Supplies LLC · 1420 Harbor Ave, Newark NJ 07102'
const BOLT_ADDR = 'Bolt & Nut Co. · 88 Industrial Park Rd, Cleveland OH 44115'
const NORTH_ADDR = 'Northwind Logistics Inc. · 2200 Commerce Blvd, Chicago IL 60616'
const CEDAR_ADDR = 'Cedar Hardware Supply · 45 Timber Lane, Portland OR 97209'
const QUANTUM_ADDR = 'Quantum Freight Systems · 700 Dockside Way, Long Beach CA 90802'

/** PO-5001 / INV-1001 — coincidencia perfecta. */
const PO_5001: Doc = {
  kind: 'PURCHASE ORDER',
  number: 'PO-5001',
  vendor: 'Acme Office Supplies LLC',
  vendorAddress: ACME_ADDR,
  date: '2026-07-02',
  currency: 'USD',
  items: [
    { description: 'A4 Copy Paper, 80gsm, ream', quantity: 40, unitPrice: 6.5 },
    { description: 'Toner Cartridge HP 26X', quantity: 8, unitPrice: 145.0 },
    { description: 'Stapler, heavy duty', quantity: 10, unitPrice: 42.0 }
  ],
  printedTotal: 1840.0
}

const INV_1001: Doc = {
  kind: 'INVOICE',
  number: 'INV-1001',
  vendor: 'Acme Office Supplies LLC',
  vendorAddress: ACME_ADDR,
  date: '2026-07-15',
  reference: 'PO-5001',
  currency: 'USD',
  items: [
    { description: 'A4 Copy Paper, 80gsm, ream', quantity: 40, unitPrice: 6.5 },
    { description: 'Toner Cartridge HP 26X', quantity: 8, unitPrice: 145.0 },
    { description: 'Stapler, heavy duty', quantity: 10, unitPrice: 42.0 }
  ],
  printedTotal: 1840.0
}

/** PO-5002 / INV-1002 — coincidencia perfecta (factura entregada como PNG). */
const PO_5002: Doc = {
  kind: 'PURCHASE ORDER',
  number: 'PO-5002',
  vendor: 'Bolt & Nut Co.',
  vendorAddress: BOLT_ADDR,
  date: '2026-07-05',
  currency: 'USD',
  items: [
    { description: 'Hex Bolt M12 x 60mm, box of 100', quantity: 25, unitPrice: 78.0 },
    { description: 'Lock Washer M12, box of 500', quantity: 15, unitPrice: 22.7 },
    { description: 'Threadlocker adhesive 50ml', quantity: 12, unitPrice: 82.5 }
  ],
  printedTotal: 3275.5
}

const INV_1002: Doc = {
  kind: 'INVOICE',
  number: 'INV-1002',
  vendor: 'Bolt & Nut Co.',
  vendorAddress: BOLT_ADDR,
  date: '2026-07-18',
  reference: 'PO-5002',
  currency: 'USD',
  items: [
    { description: 'Hex Bolt M12 x 60mm, box of 100', quantity: 25, unitPrice: 78.0 },
    { description: 'Lock Washer M12, box of 500', quantity: 15, unitPrice: 22.7 },
    { description: 'Threadlocker adhesive 50ml', quantity: 12, unitPrice: 82.5 }
  ],
  printedTotal: 3275.5
}

/** PO-5003 / INV-1003 — DISCREPANCIA: sobrefacturación de flete (+420.00). */
const PO_5003: Doc = {
  kind: 'PURCHASE ORDER',
  number: 'PO-5003',
  vendor: 'Northwind Logistics Inc.',
  vendorAddress: NORTH_ADDR,
  date: '2026-07-08',
  currency: 'USD',
  items: [
    { description: 'Container drayage, port to warehouse', quantity: 6, unitPrice: 520.0 },
    { description: 'Palletizing service', quantity: 12, unitPrice: 90.0 }
  ],
  printedTotal: 4200.0
}

const INV_1003: Doc = {
  kind: 'INVOICE',
  number: 'INV-1003',
  vendor: 'Northwind Logistics Inc.',
  vendorAddress: NORTH_ADDR,
  date: '2026-07-21',
  reference: 'PO-5003',
  currency: 'USD',
  items: [
    { description: 'Container drayage, port to warehouse', quantity: 6, unitPrice: 520.0 },
    { description: 'Palletizing service', quantity: 12, unitPrice: 90.0 },
    { description: 'Fuel surcharge', quantity: 1, unitPrice: 420.0 }
  ],
  printedTotal: 4620.0
}

/**
 * PO-5004 / INV-1004 — DISCREPANCIA: la factura omite un ítem del PO pero
 * factura el total completo (entrega parcial cobrada como completa).
 */
const PO_5004: Doc = {
  kind: 'PURCHASE ORDER',
  number: 'PO-5004',
  vendor: 'Cedar Hardware Supply',
  vendorAddress: CEDAR_ADDR,
  date: '2026-07-10',
  currency: 'USD',
  items: [
    { description: 'Circular saw blade 190mm', quantity: 20, unitPrice: 34.0 },
    { description: 'Safety goggles, polycarbonate', quantity: 60, unitPrice: 11.0 },
    { description: 'Work gloves, leather, pair', quantity: 80, unitPrice: 9.5 },
    { description: 'Cordless drill 18V', quantity: 4, unitPrice: 205.0 }
  ],
  printedTotal: 2980.0
}

const INV_1004: Doc = {
  kind: 'INVOICE',
  number: 'INV-1004',
  vendor: 'Cedar Hardware Supply',
  vendorAddress: CEDAR_ADDR,
  date: '2026-07-24',
  reference: 'PO-5004',
  currency: 'USD',
  items: [
    { description: 'Circular saw blade 190mm', quantity: 20, unitPrice: 34.0 },
    { description: 'Safety goggles, polycarbonate', quantity: 60, unitPrice: 11.0 },
    { description: 'Work gloves, leather, pair', quantity: 80, unitPrice: 9.5 }
  ],
  printedTotal: 2980.0,
  notes: 'Cordless drill 18V backordered; billed in full per contract terms.'
}

/** INV-1005 — sin orden de compra correspondiente en la carpeta de respaldo. */
const INV_1005: Doc = {
  kind: 'INVOICE',
  number: 'INV-1005',
  vendor: 'Quantum Freight Systems',
  vendorAddress: QUANTUM_ADDR,
  date: '2026-07-26',
  currency: 'USD',
  items: [
    { description: 'Expedited air freight, 3 pallets', quantity: 3, unitPrice: 1150.0 },
    { description: 'Customs brokerage fee', quantity: 1, unitPrice: 385.0 }
  ],
  printedTotal: 3835.0,
  notes: 'Emergency shipment authorized verbally by operations.'
}

/** PO-5006 / INV-1006 — DISCREPANCIA: precio unitario inflado (12.00 → 13.50). */
const PO_5006: Doc = {
  kind: 'PURCHASE ORDER',
  number: 'PO-5006',
  vendor: 'Acme Office Supplies LLC',
  vendorAddress: ACME_ADDR,
  date: '2026-07-12',
  currency: 'USD',
  items: [
    { description: 'Archive box, corrugated', quantity: 50, unitPrice: 12.0 },
    { description: 'Label sheets, 100 per pack', quantity: 30, unitPrice: 8.0 }
  ],
  printedTotal: 840.0
}

const INV_1006: Doc = {
  kind: 'INVOICE',
  number: 'INV-1006',
  vendor: 'Acme Office Supplies LLC',
  vendorAddress: ACME_ADDR,
  date: '2026-07-28',
  reference: 'PO-5006',
  currency: 'USD',
  items: [
    { description: 'Archive box, corrugated', quantity: 50, unitPrice: 13.5 },
    { description: 'Label sheets, 100 per pack', quantity: 30, unitPrice: 8.0 }
  ],
  printedTotal: 915.0
}

/** Ground truth: lo que el auditor debería concluir en cada caso. */
const EXPECTED = [
  ['INV-1001', 'PO-5001', 'MATCH', 'Ítems, cantidades y total idénticos.'],
  ['INV-1002', 'PO-5002', 'MATCH', 'Ítems, cantidades y total idénticos (factura escaneada a PNG).'],
  ['INV-1003', 'PO-5003', 'DISCREPANCY', 'Recargo de combustible de 420.00 no autorizado en el PO (4,620.00 vs 4,200.00).'],
  ['INV-1004', 'PO-5004', 'DISCREPANCY', 'Falta el ítem "Cordless drill 18V" pero se factura el total completo de 2,980.00.'],
  ['INV-1005', '—', 'UNCERTAIN', 'No existe orden de compra de respaldo; no hay evidencia para validar.'],
  ['INV-1006', 'PO-5006', 'DISCREPANCY', 'Precio unitario de "Archive box" inflado 12.00 → 13.50 (915.00 vs 840.00).']
] as const

async function main() {
  await rm(INVOICES_DIR, { recursive: true, force: true })
  await rm(SUPPORT_DIR, { recursive: true, force: true })
  await mkdir(INVOICES_DIR, { recursive: true })
  await mkdir(SUPPORT_DIR, { recursive: true })

  // Documentos de respaldo: siempre PDF.
  for (const po of [PO_5001, PO_5002, PO_5003, PO_5004, PO_5006]) {
    const out = path.join(SUPPORT_DIR, `${po.number}.pdf`)
    await renderPdf(po, out)
    console.log(`  support/  ${po.number}.pdf`)
  }

  // Facturas: mezcla de PDF nativo y PNG "escaneado" para ejercitar ambas ramas.
  const asPdf = [INV_1001, INV_1003, INV_1005, INV_1006]
  const asPng = [INV_1002, INV_1004]

  for (const inv of asPdf) {
    const out = path.join(INVOICES_DIR, `${inv.number}.pdf`)
    await renderPdf(inv, out)
    console.log(`  invoices/ ${inv.number}.pdf`)
  }

  for (const inv of asPng) {
    const tmpPdf = path.join(INVOICES_DIR, `.${inv.number}.tmp.pdf`)
    const out = path.join(INVOICES_DIR, `${inv.number}.png`)
    await renderPdf(inv, tmpPdf)
    await rasterizeToPng(tmpPdf, out)
    await rm(tmpPdf, { force: true })
    console.log(`  invoices/ ${inv.number}.png`)
  }

  // Tabla de ground truth, para contrastar contra la salida del pipeline.
  const rows = EXPECTED.map(
    ([inv, po, verdict, reason]) => `| ${inv} | ${po} | \`${verdict}\` | ${reason} |`
  ).join('\n')
  const md = `# Ground truth de los datos de prueba

Generado por \`samples/generate.ts\`. El pipeline debería reproducir estos veredictos.

| Factura | PO esperado | Veredicto esperado | Motivo |
| --- | --- | --- | --- |
${rows}
`
  await writeFile(path.join(__dirname, 'EXPECTED.md'), md, 'utf8')
  console.log('\n  samples/EXPECTED.md (ground truth)')
  console.log(`\n✔ ${asPdf.length + asPng.length} facturas y 5 órdenes de compra generadas.`)
}

main().catch((error) => {
  console.error('✖ Falló la generación de datos de prueba:', error)
  process.exit(1)
})
