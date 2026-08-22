# Reconciliador y Auditor de Facturas On-Device

Audita facturas contra sus documentos de respaldo **sin que un solo byte salga de la máquina**. Toda la extracción, la búsqueda semántica y el juicio de auditoría corren localmente con [`@qvac/sdk`](https://docs.qvac.tether.io/js-ts-sdk/) — no hay ninguna llamada a una API de inferencia remota.

Hackathon Crecimiento 2026 · Track QVAC by Tether.

## El problema

Conciliar facturas contra órdenes de compra es trabajo manual, repetitivo y lleno de datos sensibles: montos, proveedores, condiciones comerciales. Justamente el tipo de documento que una empresa no quiere subir a una API de terceros. Este reconciliador hace ese cruce en el portátil del auditor y devuelve, por factura, un veredicto accionable en menos de cinco segundos de lectura.

## Estado

| Milestone | Estado |
| --- | --- |
| **M0** — Datos de prueba sintéticos con discrepancias plantadas | ✅ |
| **M1** — Pipeline CLI de punta a punta | ✅ verificado 6/6 |
| **M2** — Backend Electron + IPC | ✅ |
| **M3** — Dashboard UI | ✅ |
| **M4** — Pulido, tipos TypeScript y documentación | ✅ |

## Instalación

Requiere **Node.js ≥ 22.17** y **npm ≥ 10.9**.

```bash
git clone <este-repo>
cd Hackaton-aleph-2026-BA
npm install
```

## Uso

```bash
# 1. Generar los datos de prueba (facturas + órdenes de compra sintéticas)
npm run samples

# 2a. Abrir la app de escritorio
npm start

# 2b. …o correr el mismo pipeline desde consola
npm run cli

# Contra carpetas propias:
npm run cli -- /ruta/a/facturas /ruta/a/respaldos

# Salida JSON, para scripting:
npm run cli -- --json
```

Chequeos rápidos, sin necesidad de descargar los modelos:

```bash
npm run verify     # 64 pruebas: rasterización, OCR, retrieval híbrido, schemas y verificación determinista
npm run typecheck
```

En la primera corrida el SDK descarga los pesos desde el registry de QVAC (~3,2 GB en total) y los cachea en `~/.qvac/models`. Las corridas siguientes arrancan sin red.

Para ver los logs del SDK durante el desarrollo, apuntá `QVAC_CONFIG_PATH` al `qvac.config.json` incluido (que ya trae `loggerConsoleOutput: true`):

```bash
QVAC_CONFIG_PATH=./qvac.config.json npm run cli
```

## Resultado verificado

Corrida real contra `samples/` en un MacBook Air, con los seis veredictos coincidiendo con el ground truth de [`samples/EXPECTED.md`](samples/EXPECTED.md):

| Factura | Veredicto | Motivo detectado |
| --- | --- | --- |
| INV-1001 | `MATCH` | — |
| INV-1002 | `MATCH` | — |
| INV-1003 | `DISCREPANCY` | Total excede el PO en 420,00 (recargo de combustible no autorizado) |
| INV-1004 | `DISCREPANCY` | Los ítems suman 2.100 contra un total de 2.920, y falta el taladro de 820,00 del PO |
| INV-1005 | `UNCERTAIN` | Sin orden de compra de respaldo; no hay evidencia |
| INV-1006 | `DISCREPANCY` | Total excede el PO en 75,00; precio unitario facturado a 13,50 contra 12,00 autorizado |

> INV-1007 (duplicado de INV-1001) se agregó después de esa corrida. Su veredicto esperado es `DISCREPANCY` con código `DUPLICATE_INVOICE`, decidido por código puro (mismo número de factura ya visto en el lote), sin pasar por el modelo de auditoría.

### Rendimiento

El costo del pipeline se reparte entre tres cosas medibles, y cada una tiene su perilla en [`src/services/tuning.ts`](src/services/tuning.ts):

| Etapa | De qué depende el costo | Perilla |
| --- | --- | --- |
| OCR — detección | Píxeles de la página | `PDF_RASTER_SCALE` × `OCR_MAG_RATIO` |
| OCR — reconocimiento | Cajas de texto detectadas | `OCR_RECOGNIZER_BATCH_SIZE` |
| LLM — decodificación | Tokens generados × recorridos de los pesos | `LLM_MAX_SLOTS`, `LLM_GENERATION_PARAMS` |

Las tres decisiones de fondo:

- **El detector no re-magnifica lo ya ampliado.** Las páginas se rasterizan a 2x desde el vector del PDF (1190x1684 px en una A4, ~28 px de alto por glifo). Ampliarlas otro 1,5x antes del detector lo hacía trabajar sobre 4,5 Mpx en vez de 2,0 para leer los mismos caracteres: interpolar un render vectorial no inventa detalle que el vector no tenga.
- **El reconocedor agrupa recortes.** Una A4 tiene entre 60 y 120 cajas de texto; reconocerlas de a una paga el costo fijo de la pasada cien veces por página.
- **La fase 3 decodifica varias facturas a la vez.** En un modelo cuantizado la decodificación está limitada por ancho de banda de memoria: emitir un token exige recorrer los 2,5 GB de pesos, se emita para una factura o para cuatro. El bucle secuencial pagaba ese recorrido por token *y* por factura. Ahora la fase 3 son dos lotes —todas las extracciones, después todas las auditorías— con una franja de código puro en el medio que resuelve, sin modelo, los casos que no necesitan auditoría (duplicados dentro del lote, órdenes de compra citadas que no existen). Cada factura conserva su prompt, su gramática y su contexto propio: lo único compartido es el paso de decodificación.

La cantidad de slots se calcula contra la RAM libre real de la máquina (`planLlm`), no se fija a ciegas: quedarse sin memoria en plena fase 3 no degrada la velocidad, tumba la corrida. Con un solo slot el comportamiento es idéntico al secuencial — y por eso el número **tiene que ser visible**: el stream de progreso informa cuántos slots se activaron y por qué. Un batcheo que degrada a un slot en silencio se ve exactamente igual que un batcheo que no funciona.

El costo real de un slot es la KV cache: 36 capas x 1024 de dimensión KV, clave y valor, son 144 KiB por token en fp16 — 576 MiB por slot de 4096. Cuatro slots así reservan 2,25 GiB **además** de los 2,5 GB de pesos, y en un portátil de 8 GB no entran. Por eso la cache va cuantizada a `q8_0` (288 MiB por slot, 1,13 GiB los cuatro): es lo que hace que el batcheo entre en la máquina objetivo. `QVAC_LLM_KV_CACHE=f16` lo desactiva; `QVAC_LLM_SLOTS=N` fuerza el paralelismo y saltea la estimación, que es la forma de medir el efecto sin depender de cuánta RAM haya libre en ese momento.

La decodificación es **determinista** (`temp: 0`, `top_k: 1`, semilla fija). Muestrear con temperatura al transcribir montos de un OCR sólo agrega la chance de desviarse del token correcto — y de fallar la validación zod, que cuesta un reintento completo. Además hace que dos corridas del mismo lote sean comparables entre sí, que es la condición para poder medir cualquier cambio.

### Cómo se mide

El reporte informa **tiempo de reloj** y el promedio por factura derivado de él. Sumar el tiempo de cada fila contaría dos veces el tramo que las facturas comparten mientras decodifican en paralelo; por eso el total de cada fila es la suma de sus etapas amortizadas, no su reloj propio.

Debajo del resumen sale la tabla que decide la próxima optimización:

```
DÓNDE SE FUE EL TIEMPO
fase                         carga trabajo descarga
OCR_LATIN                    x.x s   x.x s    x.x s
GTE_LARGE_FP16               x.x s   x.x s    x.x s
QWEN3_4B_INST_Q4_K_M         x.x s   x.x s    x.x s
(sin modelo cargado)                          x.x s
```

La separación entre **carga** y **trabajo** es la que importa: carga alta apunta a E/S (cachear los pesos, elegir un modelo más chico), trabajo alto apunta a inferencia (batchear, bajar resolución). Mirando sólo el total de la fase, las dos causas se ven idénticas. La fila sin modelo cargado es rasterización, resolución exacta de PO y verificación determinista — trabajo de CPU pura entre fases.

El OCR agrega su propio desglose entre detección y reconocimiento por el stream de progreso al terminar la fase 1, porque son dos perillas distintas.

## Cómo funciona

El pipeline corre en **tres fases secuenciales, con un solo modelo grande vivo por vez**. Ese es el requisito de diseño central: el objetivo es un portátil de ~8 GB de RAM, así que cada fase carga su modelo, hace todo su trabajo en lote y lo descarga antes de que arranque la siguiente.

```
              ┌─ Fase 1: OCR_LATIN (~98 MB) ────────────────────────┐
  PDF ─────►  │  rasterizar a PNG  →  OCR  →  texto plano           │
  PNG/JPG ►   └────────────────────────────── unloadModel ──────────┘
                                    │
              ┌─ Fase 2: GTE_LARGE_FP16 (~670 MB) ─────────────────┐
              │  embeddings de respaldos → workspace RAG            │
              │  búsqueda semántica por factura → top-3 candidatos  │
              └────────────────────────────── unloadModel ──────────┘
                                    │
              ┌─ Fase 3: QWEN3_4B_INST_Q4_K_M (~2,5 GB) ───────────┐
              │  lote 1: extracción a JSON validado con zod         │
              │  código puro: duplicados y POs inexistentes         │
              │  lote 2: auditoría contra la evidencia recuperada   │
              └────────────────────────────── unloadModel ──────────┘
                                    │
                        MATCH · DISCREPANCY · UNCERTAIN
```

Por eso la búsqueda RAG ocurre en la fase 2 y no dentro de la auditoría: recuperar durante la fase 3 exigiría tener el modelo de embeddings y el LLM cargados al mismo tiempo, que es exactamente lo que el presupuesto de memoria prohíbe.

### Recuperación híbrida: PO exacto primero, semántica como último recurso

Cuando una factura cita explícitamente una orden de compra ("PO Reference: PO-5003"), buscarla por similitud semántica es usar la herramienta equivocada: un embedding puede traer con score alto una orden *parecida* pero ajena, y comparar contra el documento equivocado produce acusaciones falsas. Una cita es un identificador, y los identificadores se resuelven por igualdad.

La fase 2 resuelve la evidencia en este orden ([`src/services/retrieval.ts`](src/services/retrieval.ts)):

1. **Coincidencia exacta.** Si la factura cita un PO, se lo busca — normalizado por mayúsculas, guiones y espacios — contra los nombres de archivo de los respaldos y contra los números de PO que aparecen en su texto OCR. Si aparece, **ese documento es la evidencia**, y a la auditoría le llega su texto OCR completo, no un fragmento RAG. Este paso es puro trabajo de strings sobre OCR ya calculado: corre sin ningún modelo cargado y no le cuesta nada al presupuesto de memoria.
2. **PO citado pero ausente.** Si la orden citada no aparece en *ningún* respaldo, la factura queda `UNCERTAIN` con motivo `PO_NOT_FOUND`: puede carecer de respaldo real, y el reporte lo dice. Deliberadamente **no** se cae a la búsqueda semántica "a ver si hay algo parecido" — un PO parecido pero equivocado es peor que ningún PO.
3. **Sin PO citado.** Recién ahí entra la búsqueda semántica de siempre (top-3), con las salvaguardas existentes de proveedor y score mínimo de recuperación.

Si todas las facturas del lote citan un PO resoluble, el modelo de embeddings ni siquiera se carga.

**La ruta principal es OCR → texto plano → LLM de texto.** No depende de que entre un modelo multimodal en memoria; el multimodal queda como mejora opcional, no como requisito.

### El modelo transcribe, el código compara

Este fue el hallazgo central del proyecto, y costó dos corridas contra datos reales.

La primera corrida devolvió **5 MATCH y 0 discrepancias**: ninguna de las tres diferencias plantadas fue detectada. La causa era el orden del schema — la gramática GBNF emite las claves en el orden declarado, y `verdict` estaba primero, así que el modelo quedaba obligado a comprometerse con un veredicto antes de haber leído un solo número.

Poner la evidencia antes del veredicto arregló dos de los tres casos. Pero la segunda corrida reveló algo peor: la factura sin orden de compra fue marcada `DISCREPANCY` con 90% de confianza, comparada contra una orden de compra **de otro proveedor** que la búsqueda semántica había traído con score 0,69. Una acusación falsa es un fallo más grave que una omisión.

El patrón detrás de los dos errores es el mismo: **cada vez que se le pide al modelo que compare, falla; cuando se le pide que transcriba, acierta.** Un LLM de 4B lee bien un OCR ruidoso, y es malo cruzando dos listas y restando dos números — que es justamente lo que el código hace de forma exacta, gratuita y auditable.

Así que el reparto quedó así:

- **El modelo transcribe** el respaldo recuperado: identificador, proveedor, total e ítems, uno por uno. Su prompt le prohíbe explícitamente juzgar y le pide dejar `discrepancies` vacío.
- **`verifyAgainstEvidence()` compara**, de forma completamente determinista:
  1. *¿La evidencia es de esta factura?* Se compara el proveedor del respaldo contra el de la factura, y la referencia de PO citada contra el documento recuperado. Si no corresponden, el veredicto es `UNCERTAIN` y **no se reporta ninguna diferencia** — comparar contra el documento equivocado sólo produce acusaciones falsas.
  2. *Totales*, con tolerancia de un centavo.
  3. *Coherencia interna*: los ítems de la factura deben sumar su propio total.
  4. *Cruce de listas de ítems*, con emparejamiento por solapamiento de palabras para que una variación de OCR no invente un faltante; sobre los ítems emparejados se comparan precio unitario y cantidad.

Las correcciones van siempre en dirección conservadora: una coincidencia puede degradarse a discrepancia o a incertidumbre, nunca al revés.

El punto 3 es lo que hace detectable el caso más difícil (INV-1004): ahí los totales **coinciden** —se factura el monto completo por una entrega parcial— y la factura se delata sola, porque sus propios ítems suman 2.100 contra un total de 2.920.

Ese mismo chequeo tiene una salvaguarda, porque compara dos números que salen del mismo OCR: si el importe de algún renglón no cierra con su cantidad por su precio unitario, el que no es confiable es el OCR y no la factura, así que no se acusa a nadie. En una corrida real, un importe mal leído había producido un desvío inexistente de 1.500.

Y en la corrida en que se agregó, este chequeo encontró **dos totales mal tipeados en los propios datos de prueba**. El generador ahora verifica su aritmética antes de escribir nada.

### Códigos de motivo estructurados

Cada discrepancia y cada veredicto no-MATCH llevan un `reasonCode` legible por máquina (`TOTAL_MISMATCH`, `INTERNAL_SUM_MISMATCH`, `UNIT_PRICE_MISMATCH`, `QUANTITY_MISMATCH`, `ITEM_NOT_ON_INVOICE`, `ITEM_NOT_AUTHORIZED`, `CURRENCY_MISMATCH`, `PO_NOT_FOUND`, `VENDOR_MISMATCH`, `NO_EVIDENCE`, `WEAK_RETRIEVAL`, `EVIDENCE_UNREADABLE`, `MISSING_CRITICAL_FIELD`, `DUPLICATE_INVOICE`, `MODEL_OUTPUT_INVALID`). Los emite exclusivamente el verificador determinista, así que "¿por qué el sistema decidió esto?" se responde sin volver a preguntarle al LLM: cada código es trazable a una regla concreta del código. La `confidence` del modelo sigue siendo cosmética — se muestra en el reporte, pero jamás participa de ninguna decisión.

Además, dos monedas distintas entre factura y respaldo hacen los montos incomparables (`CURRENCY_MISMATCH` → `UNCERTAIN`, sin conversión implícita ni acusaciones numéricas), y un mismo número de factura repetido dentro del lote marca la ocurrencia posterior como `DUPLICATE_INVOICE`, referenciando el archivo original — detección 100% en código, sin modelo.

### Manejo de incertidumbre

El prompt de auditoría instruye explícitamente al modelo a responder `UNCERTAIN` cuando no hay evidencia suficiente: si no se recuperó ningún respaldo, si el recuperado es de otro proveedor u otra orden de compra, o si el OCR quedó demasiado corrupto para comparar montos. Inventar una coincidencia es un error mucho más caro que admitir la duda, y el reporte lo dice así.

### Structured outputs deterministas

Los schemas zod de [`src/types.ts`](src/types.ts) son la única fuente de verdad: de ellos se derivan tanto los tipos de TypeScript como los JSON Schema que se le pasan al modelo en `responseFormat: { type: 'json_schema' }`. llama.cpp compila ese schema a una gramática GBNF que restringe la generación token a token, así que la forma del JSON está garantizada a nivel de gramática, no sólo por prompt.

zod es la segunda barrera, porque la gramática garantiza la forma pero no la coherencia. Si la validación falla, se reintenta **una vez** devolviéndole al modelo el error concreto de validación. Si vuelve a fallar, el documento se marca `UNCERTAIN` (o `ERROR`) con el motivo en el reporte — nunca se lanza una excepción sin capturar.

### Manejo de errores

Cada punto de falla degrada a un estado visible en el reporte en vez de tumbar la corrida: modelo que no carga, archivo ilegible, OCR vacío, JSON malformado, PO inexistente, búsqueda RAG fallida. Un documento roto marca esa fila y el lote sigue. Cada etapa se cronometra por separado y las latencias viajan en el resultado.

## La aplicación de escritorio

`npm start` compila el proceso principal y abre la ventana de Electron. El dashboard tiene selección nativa de carpetas, progreso en vivo por etapa y una tabla de auditoría donde cada fila se expande para mostrar el desglose que sostiene el veredicto.

La superficie que cruza de un proceso al otro es deliberadamente mínima. La ventana corre con `contextIsolation: true`, `nodeIntegration: false` y `sandbox: true`, así que el renderer no tiene acceso a Node ni al sistema de archivos: sólo ve las tres funciones que expone [`src/preload.ts`](src/preload.ts) — elegir una carpeta, correr el pipeline y escuchar el progreso. Los documentos y los modelos nunca salen del proceso principal; a la interfaz sólo le llegan veredictos ya calculados.

Todo el texto que llega desde el pipeline —nombres de proveedor, descripciones de ítems, resúmenes del modelo— se inserta con `textContent`, nunca con `innerHTML`. Son datos transcritos de documentos que el usuario no controla, y un PDF con markup en su texto no debería poder inyectar nada en la interfaz.

`pipeline:run` nunca rechaza: los fallos vuelven como `{ ok: false, error }`, de modo que un problema durante la corrida deja un mensaje visible en pantalla en vez de una ventana colgada.

## Modelos

Todos se resuelven por su constante exportada por el SDK, verificada contra el registry tipado del paquete y los ejemplos oficiales.

| Rol | Constante del SDK | Tamaño | Origen |
| --- | --- | --- | --- |
| OCR | `OCR_LATIN` (el detector CRAFT se deriva solo) | ~98 MB | [qvac en Hugging Face](https://huggingface.co/qvac) |
| Embeddings | `GTE_LARGE_FP16` | ~670 MB | [ChristianAzinn/gte-large-gguf](https://huggingface.co/ChristianAzinn/gte-large-gguf) |
| Extracción y auditoría | `QWEN3_4B_INST_Q4_K_M` | ~2,5 GB | [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) |

> **Nota sobre los nombres de las constantes.** No existen `GTE_LARGE` ni `QWEN3_4B_INST_Q4` con esos nombres exactos en el SDK. Los nombres reales son `GTE_LARGE_FP16` y `QWEN3_4B_INST_Q4_K_M`, verificados en `@qvac/sdk/dist/models/registry/models.d.ts` y usados tal cual en los ejemplos oficiales del repo de QVAC. El OCR tampoco requiere cargar detector y reconocedor por separado: `loadModel({ modelSrc: OCR_LATIN })` deriva el detector CRAFT automáticamente.

## Dónde se usa `@qvac/sdk`

Toda la inferencia vive en un solo archivo, [`src/services/qvacService.ts`](src/services/qvacService.ts):

| Qué | Dónde |
| --- | --- |
| Importación del SDK y las constantes de modelo | [`qvacService.ts` · imports](src/services/qvacService.ts#L20-L35) |
| Ciclo de vida load → usar → unload (`loadModel` / `unloadModel`) | `withModel()` |
| OCR (`ocr`) | `ocrDocument()` |
| Indexación RAG (`ragIngest`) y búsqueda (`ragSearch`) | `indexSupportDocuments()` |
| Extracción y auditoría (`completion` con `responseFormat`) | `completeJson()`, `extractAndAudit()` |
| Limpieza de workspace (`ragCloseWorkspace`) y cierre (`close`) | `reconcileFolders()`, `shutdown()` |

## Datos de prueba

`npm run samples` genera 7 facturas y 5 órdenes de compra con discrepancias plantadas a propósito, más [`samples/EXPECTED.md`](samples/EXPECTED.md) con el veredicto esperado de cada caso:

| Factura | Formato | Caso |
| --- | --- | --- |
| INV-1001 | PDF | coincidencia exacta |
| INV-1002 | PNG | coincidencia exacta (factura escaneada) |
| INV-1003 | PDF | recargo de combustible de 420,00 no autorizado en el PO |
| INV-1004 | PNG | falta un ítem del PO (820,00) pero se factura el total completo de 2.920,00 |
| INV-1005 | PDF | sin orden de compra de respaldo → debe dar `UNCERTAIN` |
| INV-1006 | PDF | precio unitario inflado de 12,00 a 13,50 |
| INV-1007 | PDF | reenvío duplicado de INV-1001 (mismo número de factura) → `DUPLICATE_INVOICE` |

Las facturas vienen en PDF y en PNG a propósito, para que cada corrida ejercite tanto la rama de rasterización como la de imagen directa.

## Estructura

```
src/
  types.ts                  schemas zod + JSON Schema para structured outputs
  cli.ts                    entry point del M1
  selftest.ts               verificación offline (npm run verify)
  main.ts                   proceso principal de Electron e IPC
  preload.ts                puente con context isolation
  services/
    qvacService.ts          toda la inferencia @qvac/sdk
    rasterize.ts            PDF → PNG con pdf-to-img
ui/
  index.html  styles.css  app.js    dashboard, sin frameworks
samples/
  generate.ts               generador de datos de prueba
  invoices/  support/       documentos generados
  EXPECTED.md               ground truth
qvac.config.json            config del SDK (loggerConsoleOutput activo)
```

## Hardware

Pensado para un portátil de ~8 GB de RAM. El pico de memoria lo fija la fase 3 (~2,5 GB del LLM más el contexto), porque las fases nunca se solapan. Corre en CPU; si hay GPU disponible el SDK la aprovecha.

## M4 — Completado

**Compilación TypeScript y desktop app funcional.**

- Instalados tipos TypeScript para Electron (`@types/electron`).
- `npm run build` compila exitosamente main.ts y preload.ts sin errores de tipo.
- `npm start` abre la app de Electron con la interfaz de auditoría lista.
- Todos los tests de verificación pasan (30/30).
- Contexto aislado confirmado: renderer sin acceso a Node ni sistema de archivos.
- Todo el código está tipado y en TypeScript estricto.
