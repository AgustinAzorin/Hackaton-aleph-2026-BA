/**
 * M1 — Pipeline completo desde consola.
 *
 * Uso:
 *   npm run cli                                   # corre contra samples/
 *   npm run cli -- <facturas> <respaldos>         # carpetas propias
 *   npm run cli -- --json                         # salida JSON para scripting
 *
 * Toda la inferencia ocurre en la máquina local a través de @qvac/sdk.
 */
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import type { ReconciliationVerdict, Verdict } from './types.js'
import { reconcileFolders, shutdown, type ProgressEvent } from './services/qvacService.js'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const REPO_ROOT = path.resolve(__dirname, '..')

const COLORS = {
  reset: '\u001b[0m',
  dim: '\u001b[2m',
  bold: '\u001b[1m',
  green: '\u001b[32m',
  red: '\u001b[31m',
  yellow: '\u001b[33m',
  cyan: '\u001b[36m'
} as const

const useColor = process.stdout.isTTY === true && process.env.NO_COLOR === undefined
const paint = (color: keyof typeof COLORS, text: string) =>
  useColor ? `${COLORS[color]}${text}${COLORS.reset}` : text

function badge(verdict: Verdict): string {
  switch (verdict) {
    case 'MATCH':
      return paint('green', '✔ MATCH      ')
    case 'DISCREPANCY':
      return paint('red', '✖ DISCREPANCY')
    case 'UNCERTAIN':
      return paint('yellow', '? UNCERTAIN  ')
  }
}

function formatProgress(event: ProgressEvent): string {
  const counter =
    event.current !== undefined && event.total !== undefined
      ? ` (${Math.round(event.current)}/${Math.round(event.total)})`
      : ''
  const timestamp = new Date().toISOString().slice(11, 19)
  return paint('dim', `[${timestamp}] ${event.stage.padEnd(7)} ${event.message}${counter}`)
}

function printReport(verdicts: ReconciliationVerdict[]): void {
  console.log(`\n${paint('bold', '═'.repeat(78))}`)
  console.log(paint('bold', 'REPORTE DE RECONCILIACIÓN'))
  console.log(paint('bold', '═'.repeat(78)))

  for (const row of verdicts) {
    console.log()

    if (row.status === 'ERROR') {
      console.log(`${paint('red', '⚠ ERROR      ')}  ${paint('bold', row.file)}`)
      console.log(`               ${row.error ?? 'Motivo desconocido.'}`)
      console.log(paint('dim', `               ${row.totalMs} ms`))
      continue
    }

    const audit = row.audit
    const invoice = row.invoice
    const header = audit ? badge(audit.verdict) : paint('yellow', '? UNCERTAIN  ')
    console.log(`${header}  ${paint('bold', row.file)}`)

    if (invoice) {
      const total = `${invoice.currency} ${invoice.totalAmount.toFixed(2)}`
      const po = invoice.poReference.length > 0 ? invoice.poReference : '—'
      console.log(
        `               ${invoice.invoiceNumber} · ${invoice.vendorName} · ${total} · PO ${po}`
      )
    }

    if (audit) {
      console.log(
        `               ${audit.summary} ${paint('dim', `(confianza ${(audit.confidence * 100).toFixed(0)}%)`)}`
      )
      for (const d of audit.discrepancies) {
        console.log(
          `                 ${paint('cyan', '·')} ${d.field}: factura ${d.invoiceValue} vs respaldo ${d.supportValue} — ${d.difference}`
        )
      }
    }

    const support =
      row.matchedSupportDoc !== null
        ? `respaldo ${row.matchedSupportDoc}${row.supportScore !== null ? ` (score ${row.supportScore.toFixed(3)})` : ''}`
        : 'sin respaldo recuperado'
    const stages = row.timings.map((t) => `${t.stage} ${t.ms}ms`).join(' · ')
    console.log(paint('dim', `               ${support} · ${stages} · total ${row.totalMs} ms`))
  }

  // Resumen agregado
  const counts = { MATCH: 0, DISCREPANCY: 0, UNCERTAIN: 0, ERROR: 0 }
  for (const row of verdicts) {
    if (row.status === 'ERROR') counts.ERROR++
    else if (row.audit) counts[row.audit.verdict]++
    else counts.UNCERTAIN++
  }

  const totalMs = verdicts.reduce((sum, r) => sum + r.totalMs, 0)
  console.log(`\n${paint('bold', '─'.repeat(78))}`)
  console.log(
    `${verdicts.length} facturas · ` +
      `${paint('green', `${counts.MATCH} match`)} · ` +
      `${paint('red', `${counts.DISCREPANCY} discrepancias`)} · ` +
      `${paint('yellow', `${counts.UNCERTAIN} inciertas`)} · ` +
      `${counts.ERROR} errores · ${(totalMs / 1000).toFixed(1)} s de proceso`
  )
  console.log(paint('bold', '─'.repeat(78)))
}

async function main(): Promise<void> {
  const args = process.argv.slice(2)
  const asJson = args.includes('--json')
  const positional = args.filter((a) => !a.startsWith('--'))

  const invoicesDir = path.resolve(positional[0] ?? path.join(REPO_ROOT, 'samples/invoices'))
  const supportDir = path.resolve(positional[1] ?? path.join(REPO_ROOT, 'samples/support'))

  if (!asJson) {
    console.log(paint('bold', '\nReconciliador de Facturas On-Device · QVAC'))
    console.log(paint('dim', `Facturas:  ${invoicesDir}`))
    console.log(paint('dim', `Respaldos: ${supportDir}`))
    console.log(paint('dim', 'Toda la inferencia corre localmente vía @qvac/sdk.\n'))
  }

  const started = Date.now()
  const verdicts = await reconcileFolders({
    invoicesDir,
    supportDir,
    onProgress: (event) => {
      if (!asJson) console.error(formatProgress(event))
    }
  })

  if (asJson) {
    console.log(JSON.stringify({ verdicts, elapsedMs: Date.now() - started }, null, 2))
  } else {
    printReport(verdicts)
  }
}

main()
  .then(async () => {
    await shutdown()
    process.exit(0)
  })
  .catch(async (error) => {
    // Último cortafuegos: la app no debe terminar con una excepción sin capturar.
    console.error(
      `\n${paint('red', '✖ El pipeline falló:')} ${error instanceof Error ? error.message : String(error)}`
    )
    if (error instanceof Error && error.stack !== undefined) {
      console.error(paint('dim', error.stack.split('\n').slice(1).join('\n')))
    }
    await shutdown()
    process.exit(1)
  })
