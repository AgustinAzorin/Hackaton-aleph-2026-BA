/**
 * Puente seguro entre el proceso principal y la interfaz.
 *
 * La ventana corre con `contextIsolation: true` y `nodeIntegration: false`, así
 * que el renderer no tiene acceso a Node ni al sistema de archivos. Lo único
 * que ve es la superficie mínima que se expone acá: elegir carpetas, disparar
 * el pipeline y escuchar el progreso. Nada más cruza la frontera.
 *
 * Se compila a CommonJS (`dist/preload.cjs`) porque los preload de Electron
 * dentro del sandbox no aceptan módulos ES.
 */
import { contextBridge, ipcRenderer, type IpcRendererEvent } from 'electron'
import type { ProgressEvent } from './services/qvacService.js'
import type { PhaseTiming, ReconciliationVerdict } from './types.js'

export interface PipelineRequest {
  invoicesDir: string
  supportDir: string
}

export type PipelineResponse =
  | { ok: true; verdicts: ReconciliationVerdict[]; phases: PhaseTiming[]; elapsedMs: number }
  | { ok: false; error: string }

export interface ReconcilerApi {
  /** Abre el selector nativo de directorios. `null` si el usuario cancela. */
  selectFolder: (title: string) => Promise<string | null>
  /** Corre el pipeline completo. Nunca rechaza: los fallos vuelven en `ok: false`. */
  runPipeline: (request: PipelineRequest) => Promise<PipelineResponse>
  /** Se suscribe al progreso en vivo. Devuelve la función para desuscribirse. */
  onProgress: (listener: (event: ProgressEvent) => void) => () => void
}

const api: ReconcilerApi = {
  selectFolder: (title) => ipcRenderer.invoke('dialog:selectFolder', title),
  runPipeline: (request) => ipcRenderer.invoke('pipeline:run', request),
  onProgress: (listener) => {
    // El evento de IPC no se le pasa al consumidor: expone `sender`, y con él
    // una vía de vuelta al proceso principal que la interfaz no necesita.
    const handler = (_event: IpcRendererEvent, progress: ProgressEvent) => listener(progress)
    ipcRenderer.on('pipeline:progress', handler)
    return () => ipcRenderer.removeListener('pipeline:progress', handler)
  }
}

contextBridge.exposeInMainWorld('reconciler', api)
