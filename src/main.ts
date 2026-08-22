/**
 * M2 — Proceso principal de Electron.
 *
 * Envuelve el pipeline del M1 detrás de IPC. El pipeline no cambia: es el mismo
 * `reconcileFolders()` que corre la CLI, con su callback de progreso conectado
 * al renderer para que la interfaz muestre las etapas en vivo.
 *
 * Toda la inferencia sigue ocurriendo en el proceso principal, en la máquina
 * local. El renderer nunca ve un archivo ni un modelo: sólo recibe veredictos.
 */
import { app, BrowserWindow, dialog, ipcMain, shell } from 'electron'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { reconcileFolders, shutdown, type ProgressEvent } from './services/qvacService.js'
import type { PipelineRequest, PipelineResponse } from './preload.js'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const APP_ROOT = path.resolve(__dirname, '..')

let mainWindow: BrowserWindow | null = null

/** Impide que dos corridas se pisen: los modelos no entran dos veces en RAM. */
let pipelineBusy = false

function createWindow(): void {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 860,
    minWidth: 960,
    minHeight: 600,
    backgroundColor: '#0f1115',
    title: 'Reconciliador de Facturas · QVAC',
    show: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true
    }
  })

  // Evita el destello blanco mientras carga la hoja de estilos oscura.
  mainWindow.once('ready-to-show', () => mainWindow?.show())
  void mainWindow.loadFile(path.join(APP_ROOT, 'ui/index.html'))

  // La app es completamente local: cualquier navegación externa sale al
  // navegador del sistema en vez de secuestrar la ventana.
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    void shell.openExternal(url)
    return { action: 'deny' }
  })

  mainWindow.on('closed', () => {
    mainWindow = null
  })
}

// ---------------------------------------------------------------------------
// IPC
// ---------------------------------------------------------------------------

ipcMain.handle('dialog:selectFolder', async (_event, title: string) => {
  if (mainWindow === null) return null

  const result = await dialog.showOpenDialog(mainWindow, {
    title,
    properties: ['openDirectory', 'createDirectory']
  })

  return result.canceled ? null : (result.filePaths[0] ?? null)
})

ipcMain.handle(
  'pipeline:run',
  async (event, request: PipelineRequest): Promise<PipelineResponse> => {
    if (pipelineBusy) {
      return { ok: false, error: 'Ya hay una reconciliación en curso.' }
    }

    if (
      typeof request?.invoicesDir !== 'string' ||
      typeof request?.supportDir !== 'string' ||
      request.invoicesDir.length === 0 ||
      request.supportDir.length === 0
    ) {
      return { ok: false, error: 'Faltan las carpetas de facturas o de respaldos.' }
    }

    pipelineBusy = true
    const started = Date.now()

    try {
      const verdicts = await reconcileFolders({
        invoicesDir: request.invoicesDir,
        supportDir: request.supportDir,
        onProgress: (progress: ProgressEvent) => {
          // La ventana puede haberse cerrado en medio de una corrida larga.
          if (!event.sender.isDestroyed()) {
            event.sender.send('pipeline:progress', progress)
          }
        }
      })

      return { ok: true, verdicts, elapsedMs: Date.now() - started }
    } catch (error) {
      // El pipeline degrada cada documento por su cuenta, así que llegar acá
      // significa un fallo de la corrida entera (carpeta ilegible, modelo que
      // no carga). Vuelve como dato, no como excepción: la ventana no se cae.
      return { ok: false, error: error instanceof Error ? error.message : String(error) }
    } finally {
      pipelineBusy = false
    }
  }
)

// ---------------------------------------------------------------------------
// Ciclo de vida
// ---------------------------------------------------------------------------

void app.whenReady().then(() => {
  createWindow()

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

// Cierra la conexión RPC del SDK para que no queden procesos worker huérfanos.
app.on('before-quit', () => {
  void shutdown()
})
