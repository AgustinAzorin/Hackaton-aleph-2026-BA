/**
 * M3 — Dashboard de auditoría.
 *
 * El renderer no tiene acceso a Node ni al sistema de archivos: todo pasa por
 * la superficie mínima que expone el preload en `window.reconciler`.
 *
 * Todo el texto que llega desde el pipeline (nombres de proveedor, descripciones
 * de ítems, resúmenes del modelo) se inserta con `textContent`, nunca con
 * `innerHTML`. Son datos transcritos de documentos que el usuario no controla,
 * y un PDF con markup en su texto no debe poder inyectar nada en la interfaz.
 */

const api = window.reconciler

const el = (id) => document.getElementById(id)

const state = {
  invoicesDir: null,
  supportDir: null,
  running: false,
  startedAt: 0,
  timer: null
}

// ---------------------------------------------------------------------------
// Formato
// ---------------------------------------------------------------------------

const seconds = (ms) => `${(ms / 1000).toFixed(1).replace('.', ',')} s`

const money = (amount, currency) =>
  `${currency} ${amount.toLocaleString('es-AR', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  })}`

const VERDICT_LABEL = {
  MATCH: 'MATCH',
  DISCREPANCY: 'DISCREPANCIA',
  UNCERTAIN: 'INCIERTO',
  ERROR: 'ERROR'
}

/** Crea un elemento con texto seguro y clases opcionales. */
function node(tag, className, text) {
  const element = document.createElement(tag)
  if (className) element.className = className
  if (text !== undefined) element.textContent = text
  return element
}

// ---------------------------------------------------------------------------
// Selección de carpetas
// ---------------------------------------------------------------------------

function setPath(target, dir) {
  const field = el(`path-${target}`)
  if (dir) {
    field.textContent = dir
    field.dataset.empty = 'false'
    field.title = dir
  } else {
    field.textContent = 'Ninguna carpeta elegida'
    field.dataset.empty = 'true'
    field.removeAttribute('title')
  }
  el('run').disabled = state.running || !state.invoicesDir || !state.supportDir
}

el('pick-invoices').addEventListener('click', async () => {
  const dir = await api.selectFolder('Elegí la carpeta con las facturas')
  if (dir) {
    state.invoicesDir = dir
    setPath('invoices', dir)
  }
})

el('pick-support').addEventListener('click', async () => {
  const dir = await api.selectFolder('Elegí la carpeta con los documentos de respaldo')
  if (dir) {
    state.supportDir = dir
    setPath('support', dir)
  }
})

// ---------------------------------------------------------------------------
// Progreso en vivo
// ---------------------------------------------------------------------------

const STAGE_LABEL = {
  model: 'Modelos',
  ocr: 'Fase 1/3 · OCR',
  index: 'Fase 2/3 · Indexación RAG',
  extract: 'Fase 3/3 · Extracción',
  audit: 'Fase 3/3 · Auditoría',
  done: 'Completo'
}

function appendLog(event) {
  const log = el('log')
  const line = node('li')
  const now = new Date()
  const stamp = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}`

  line.appendChild(node('span', 'ts', stamp))
  line.appendChild(document.createTextNode(event.message))
  log.appendChild(line)

  // Sólo se autoscrollea si el usuario no se fue a leer hacia arriba.
  const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 40
  if (atBottom) log.scrollTop = log.scrollHeight

  while (log.children.length > 400) log.removeChild(log.firstChild)
}

function handleProgress(event) {
  el('stage').textContent = STAGE_LABEL[event.stage] ?? event.stage
  appendLog(event)

  const fill = el('bar-fill')
  if (typeof event.current === 'number' && typeof event.total === 'number' && event.total > 0) {
    fill.classList.remove('indeterminate')
    fill.style.width = `${Math.min(100, (event.current / event.total) * 100)}%`
  } else {
    // Sin numerador conocido la barra late, en vez de fingir un progreso.
    fill.classList.add('indeterminate')
    fill.style.width = ''
  }
}

function startTimer() {
  state.startedAt = Date.now()
  state.timer = setInterval(() => {
    el('elapsed').textContent = seconds(Date.now() - state.startedAt)
  }, 100)
}

function stopTimer() {
  if (state.timer !== null) clearInterval(state.timer)
  state.timer = null
}

// ---------------------------------------------------------------------------
// Tabla de resultados
// ---------------------------------------------------------------------------

/** Fila expandible con el desglose auditable de una factura. */
function renderRow(row, index) {
  const verdict = row.status === 'ERROR' ? 'ERROR' : (row.audit?.verdict ?? 'UNCERTAIN')
  const detailId = `detail-${index}`

  const tr = node('tr', 'row-main')
  tr.setAttribute('role', 'button')
  tr.setAttribute('tabindex', '0')
  tr.setAttribute('aria-expanded', 'false')
  tr.setAttribute('aria-controls', detailId)

  // Veredicto
  const tdVerdict = node('td')
  const badge = node('span', `verdict verdict-${verdict}`)
  badge.appendChild(node('span', 'chevron', '›'))
  badge.appendChild(document.createTextNode(VERDICT_LABEL[verdict]))
  tdVerdict.appendChild(badge)
  tr.appendChild(tdVerdict)

  // Archivo + motivo en una línea: es lo que el auditor lee primero.
  const tdFile = node('td')
  tdFile.appendChild(node('span', 'file', row.file))
  const reason =
    row.status === 'ERROR'
      ? (row.error ?? 'Motivo desconocido.')
      : (row.audit?.summary ?? 'Sin veredicto.')
  tdFile.appendChild(node('span', 'reason', reason))
  tr.appendChild(tdFile)

  // Proveedor y total
  tr.appendChild(node('td', undefined, row.invoice?.vendorName ?? '—'))
  tr.appendChild(
    node(
      'td',
      'num',
      row.invoice ? money(row.invoice.totalAmount, row.invoice.currency) : '—'
    )
  )

  // Respaldo recuperado
  const tdSupport = node('td', 'support')
  if (row.matchedSupportDoc) {
    tdSupport.appendChild(document.createTextNode(row.matchedSupportDoc))
    if (typeof row.supportScore === 'number') {
      tdSupport.appendChild(
        node('span', 'score', `  ${row.supportScore.toFixed(2).replace('.', ',')}`)
      )
    }
  } else {
    tdSupport.textContent = 'sin respaldo'
  }
  tr.appendChild(tdSupport)

  tr.appendChild(node('td', 'num latency', seconds(row.totalMs)))

  // --- Desglose ---
  const detail = node('tr', 'detail')
  detail.id = detailId
  detail.hidden = true

  const detailCell = node('td')
  detailCell.colSpan = 6
  const grid = node('div', 'detail-grid')

  const discrepancies = row.audit?.discrepancies ?? []
  if (discrepancies.length > 0) {
    for (const d of discrepancies) {
      const box = node('div', 'diff')
      box.appendChild(node('span', 'diff-field', d.field))

      const values = node('span', 'diff-values')
      values.appendChild(node('span', 'inv', d.invoiceValue))
      values.appendChild(document.createTextNode('  vs  '))
      values.appendChild(node('span', 'sup', d.supportValue))
      box.appendChild(values)

      box.appendChild(node('span'))
      box.appendChild(node('span', 'diff-note', d.difference))
      grid.appendChild(box)
    }
  } else {
    grid.appendChild(
      node(
        'div',
        'no-diff',
        row.status === 'ERROR'
          ? 'El pipeline no llegó a auditar este documento.'
          : 'No se encontraron diferencias contra el documento de respaldo.'
      )
    )
  }

  const timings = row.timings.map((t) => `${t.stage} ${seconds(t.ms)}`).join('  ·  ')
  grid.appendChild(node('div', 'timings', `${timings}  ·  total ${seconds(row.totalMs)}`))

  detailCell.appendChild(grid)
  detail.appendChild(detailCell)

  const toggle = () => {
    const open = tr.getAttribute('aria-expanded') === 'true'
    tr.setAttribute('aria-expanded', String(!open))
    detail.hidden = open
  }

  tr.addEventListener('click', toggle)
  tr.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      toggle()
    }
  })

  return [tr, detail]
}

function renderResults(verdicts, elapsedMs) {
  const counts = { MATCH: 0, DISCREPANCY: 0, UNCERTAIN: 0, ERROR: 0 }
  const tbody = el('rows')
  tbody.replaceChildren()

  verdicts.forEach((row, index) => {
    const verdict = row.status === 'ERROR' ? 'ERROR' : (row.audit?.verdict ?? 'UNCERTAIN')
    counts[verdict]++
    for (const element of renderRow(row, index)) tbody.appendChild(element)
  })

  el('count-match').textContent = String(counts.MATCH)
  el('count-discrepancy').textContent = String(counts.DISCREPANCY)
  el('count-uncertain').textContent = String(counts.UNCERTAIN)
  el('count-error').textContent = String(counts.ERROR)
  el('count-time').textContent = seconds(elapsedMs)

  el('summary').hidden = false
  el('results').hidden = false
  el('empty').hidden = true
}

// ---------------------------------------------------------------------------
// Corrida
// ---------------------------------------------------------------------------

api.onProgress(handleProgress)

el('run').addEventListener('click', async () => {
  if (state.running) return

  state.running = true
  el('run').disabled = true
  el('pick-invoices').disabled = true
  el('pick-support').disabled = true
  el('error-banner').hidden = true
  el('log').replaceChildren()
  el('progress').hidden = false
  el('bar-fill').classList.add('indeterminate')
  el('stage').textContent = 'Iniciando…'
  startTimer()

  // `runPipeline` no rechaza: los fallos vuelven como `ok: false`, así que la
  // interfaz nunca queda colgada con el botón deshabilitado.
  const response = await api.runPipeline({
    invoicesDir: state.invoicesDir,
    supportDir: state.supportDir
  })

  stopTimer()
  state.running = false
  el('pick-invoices').disabled = false
  el('pick-support').disabled = false
  el('run').disabled = false
  el('bar-fill').classList.remove('indeterminate')

  if (response.ok) {
    el('bar-fill').style.width = '100%'
    el('stage').textContent = 'Completo'
    el('elapsed').textContent = seconds(response.elapsedMs)
    renderResults(response.verdicts, response.elapsedMs)
  } else {
    el('bar-fill').style.width = '0'
    el('stage').textContent = 'Falló'
    const banner = el('error-banner')
    banner.textContent = response.error
    banner.hidden = false
  }
})
