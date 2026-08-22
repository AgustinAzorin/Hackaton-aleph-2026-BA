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
| M4 — Pulido y métricas | pendiente |

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
npm run verify     # 30 pruebas: rasterización, OCR, RAG, schemas y verificación determinista
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

Alrededor de 30 segundos por factura de punta a punta (OCR, extracción y auditoría), con los tres modelos cargándose y descargándose por fase.

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
              │  extracción a JSON validado con zod                 │
              │  auditoría contra la evidencia recuperada           │
              └────────────────────────────── unloadModel ──────────┘
                                    │
                        MATCH · DISCREPANCY · UNCERTAIN
```

Por eso la búsqueda RAG ocurre en la fase 2 y no dentro de la auditoría: recuperar durante la fase 3 exigiría tener el modelo de embeddings y el LLM cargados al mismo tiempo, que es exactamente lo que el presupuesto de memoria prohíbe.

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

`npm run samples` genera 6 facturas y 5 órdenes de compra con discrepancias plantadas a propósito, más [`samples/EXPECTED.md`](samples/EXPECTED.md) con el veredicto esperado de cada caso:

| Factura | Formato | Caso |
| --- | --- | --- |
| INV-1001 | PDF | coincidencia exacta |
| INV-1002 | PNG | coincidencia exacta (factura escaneada) |
| INV-1003 | PDF | recargo de combustible de 420,00 no autorizado en el PO |
| INV-1004 | PNG | falta un ítem del PO (820,00) pero se factura el total completo de 2.920,00 |
| INV-1005 | PDF | sin orden de compra de respaldo → debe dar `UNCERTAIN` |
| INV-1006 | PDF | precio unitario inflado de 12,00 a 13,50 |

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
