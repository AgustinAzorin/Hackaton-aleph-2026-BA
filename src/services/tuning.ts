/**
 * Presupuesto de rendimiento del pipeline.
 *
 * Todas las perillas que deciden *cuánto* trabajo hace cada modelo viven acá,
 * en un solo lugar, para que ajustar la velocidad no obligue a tocar la lógica
 * de negocio. Cada constante documenta qué compra y qué arriesga.
 *
 * La regla que ordena el archivo: **nada de lo que hay acá puede cambiar el
 * veredicto de una factura**. Lo que se optimiza es el camino, no el destino:
 * menos píxeles redundantes, menos llamadas al modelo, decodificación
 * determinista. Las decisiones siguen saliendo del verificador determinista de
 * `qvacService.ts`.
 */
import os from 'node:os'

// ---------------------------------------------------------------------------
// Fase 1 — OCR
// ---------------------------------------------------------------------------

/**
 * Factor de rasterización de PDF a PNG.
 *
 * Una A4 mide 595x842 pt, así que a 2x sale una imagen de 1190x1684 px y el
 * cuerpo de texto de 10 pt queda con glifos de ~28 px de alto — el rango donde
 * el reconocedor de EasyOCR trabaja mejor. Las facturas escaneadas de prueba
 * (PNG) ya vienen a esa misma resolución, así que todo el lote entra al OCR
 * con la misma densidad.
 */
export const PDF_RASTER_SCALE = 2

/**
 * Magnificación adicional que aplica el detector antes de buscar cajas de texto.
 *
 * Estaba en 1.5, encima del 2x de rasterización: el detector CRAFT terminaba
 * corriendo sobre ~4,5 Mpx (1785x2526) para leer un documento que ya estaba
 * nítido a 2,0 Mpx. El costo del detector es lineal en píxeles, así que ese 1.5
 * multiplicaba por ~2,25 la etapa más cara del OCR sin agregar información:
 * interpolar un render vectorial no inventa detalle que el vector no tenga.
 *
 * A 1.0 el detector ve exactamente los píxeles que el rasterizador produjo.
 * Si algún día entran escaneos de baja resolución (< 1000 px de lado largo),
 * subir esto es la perilla correcta — por eso queda como constante y no
 * enterrado en la llamada.
 */
export const OCR_MAG_RATIO = 1.0

/**
 * Cuántos recortes de texto reconoce el modelo por pasada.
 *
 * Estaba en 1: cada caja detectada disparaba una inferencia propia, pagando el
 * costo fijo de la pasada por cada renglón. Una factura A4 tiene entre 60 y 120
 * cajas, así que era el costo fijo multiplicado por cien. En lotes de 16 el
 * reconocedor amortiza ese costo sobre varias cajas a la vez; el resultado por
 * caja es el mismo, porque cada recorte se normaliza y se infiere de forma
 * independiente.
 */
export const OCR_RECOGNIZER_BATCH_SIZE = 16

/** Hilos del OCR: todos los núcleos disponibles, con un piso de 4. */
export function ocrThreads(): number {
  return Math.max(4, os.availableParallelism?.() ?? os.cpus().length)
}

// ---------------------------------------------------------------------------
// Fase 3 — LLM
// ---------------------------------------------------------------------------

/**
 * Contexto por factura (por *slot* de decodificación, no del proceso).
 *
 * El presupuesto real por llamada: el prompt más largo es el de auditoría —
 * system (~600 tokens) + factura extraída en JSON (~400) + respaldo recortado a
 * 6000 caracteres (~1800) ≈ 2800 tokens de entrada, y la salida más larga
 * observada ronda los 700. 4096 deja margen sobre eso; los 8192 anteriores
 * reservaban KV cache que nunca se usaba.
 */
export const LLM_CTX_PER_SLOT = 4096

/**
 * Techo de tokens generados por respuesta.
 *
 * La gramática GBNF ya cierra el JSON solo, así que en la práctica nunca se
 * llega. Es un cortafuegos contra el caso patológico —un modelo que entra en
 * bucle dentro de un array y consume la ventana entera— que en un lote
 * secuencial se llevaba puestos varios minutos sin producir nada útil.
 */
export const LLM_MAX_PREDICT = 1024

/**
 * Parámetros de decodificación: greedy y reproducible.
 *
 * El default de llama.cpp muestrea con temperatura ~0,8. Para transcribir
 * montos de un OCR eso es puro riesgo: la respuesta correcta ya es el token más
 * probable, y muestrear sólo agrega la chance de desviarse — y de fallar la
 * validación zod, que cuesta un reintento completo. Con `temp: 0` y `top_k: 1`
 * la decodificación es determinista: la misma factura devuelve el mismo JSON en
 * cada corrida, que además es lo que hace comparables dos mediciones.
 *
 * `reasoning_budget: 0` apaga el canal de razonamiento de Qwen3 de forma
 * explícita, en vez de confiar en el `/no_think` del prompt.
 */
export const LLM_GENERATION_PARAMS = {
  temp: 0,
  top_k: 1,
  seed: 1,
  predict: LLM_MAX_PREDICT,
  reasoning_budget: 0
} as const

/**
 * Cuántas facturas decodifica el LLM a la vez.
 *
 * En un modelo cuantizado la decodificación está limitada por ancho de banda de
 * memoria, no por cómputo: para producir UN token hay que recorrer los 2,5 GB de
 * pesos igual. Decodificar cuatro secuencias en paralelo recorre esos mismos
 * pesos una sola vez y produce cuatro tokens, así que el costo por factura cae
 * casi en proporción al número de slots. Es la razón por la que la fase 3 pasa
 * de un bucle secuencial a dos lotes (extracción y auditoría).
 *
 * El límite es la KV cache: cada slot reserva `LLM_CTX_PER_SLOT` tokens, y en
 * Qwen3-4B eso son ~150 MB por slot en fp16. Con el modelo ocupando 2,5 GB, el
 * número de slots se calcula contra la RAM libre real en vez de fijarse a ciegas.
 */
export const LLM_MAX_SLOTS = 4

/** Peso del modelo en RAM, para descontarlo del presupuesto. */
const LLM_WEIGHTS_BYTES = 2_500_000_000

/** KV cache aproximada de un slot de `LLM_CTX_PER_SLOT` tokens en Qwen3-4B. */
const LLM_KV_BYTES_PER_SLOT = 160_000_000

/** Margen que se le deja al sistema operativo y al resto de la app. */
const HEADROOM_BYTES = 1_200_000_000

export interface LlmPlan {
  /** Slots de decodificación concurrentes. */
  slots: number
  /** Contexto total del proceso: se reparte entre los slots. */
  ctxSize: number
}

/**
 * Elige cuántos slots caben en la máquina actual.
 *
 * Se mide contra la memoria **libre** y no contra la total: un portátil de
 * 8 GB con el navegador abierto tiene mucho menos disponible que 8 GB, y
 * quedarse sin RAM en la fase 3 no degrada la velocidad, tumba la corrida.
 * Con un solo slot el comportamiento es exactamente el de antes.
 */
export function planLlm(invoiceCount: number): LlmPlan {
  const available = Math.max(os.freemem(), os.totalmem() * 0.5)
  const forKv = available - LLM_WEIGHTS_BYTES - HEADROOM_BYTES
  const affordable = Math.floor(forKv / LLM_KV_BYTES_PER_SLOT)

  const slots = Math.max(1, Math.min(LLM_MAX_SLOTS, affordable, Math.max(1, invoiceCount)))
  return { slots, ctxSize: slots * LLM_CTX_PER_SLOT }
}
