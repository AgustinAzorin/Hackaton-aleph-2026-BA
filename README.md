# Reconciliador y Auditor de Facturas On-Device

Audita facturas contra sus documentos de respaldo **sin que un solo byte salga de la máquina**. Toda la extracción, la búsqueda semántica y el juicio de auditoría corren localmente con [`@qvac/sdk`](https://docs.qvac.tether.io/js-ts-sdk/) — no hay ninguna llamada a una API de inferencia remota.

Hackathon Crecimiento 2026 · Track QVAC by Tether.

## El problema

Conciliar facturas contra órdenes de compra es trabajo manual, repetitivo y lleno de datos sensibles: montos, proveedores, condiciones comerciales. Justamente el tipo de documento que una empresa no quiere subir a una API de terceros. Este reconciliador hace ese cruce en el portátil del auditor y devuelve, por factura, un veredicto accionable en menos de cinco segundos de lectura.

## Estado

| Milestone | Estado |
| --- | --- |
| **M0** — Datos de prueba sintéticos con discrepancias plantadas | ✅ |
| **M1** — Pipeline CLI de punta a punta | ✅ |
| M2 — Backend Electron + IPC | pendiente |
| M3 — Dashboard UI | pendiente |
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

# 2. Correr el pipeline completo contra samples/
npm run cli

# Contra carpetas propias:
npm run cli -- /ruta/a/facturas /ruta/a/respaldos

# Salida JSON, para scripting:
npm run cli -- --json
```

Chequeos rápidos, sin necesidad de descargar los modelos:

```bash
npm run verify     # 28 pruebas: rasterización, OCR, RAG, schemas y verificación determinista
npm run typecheck
```

En la primera corrida el SDK descarga los pesos desde el registry de QVAC (~3,2 GB en total) y los cachea en `~/.qvac/models`. Las corridas siguientes arrancan sin red.

Para ver los logs del SDK durante el desarrollo, apuntá `QVAC_CONFIG_PATH` al `qvac.config.json` incluido (que ya trae `loggerConsoleOutput: true`):

```bash
QVAC_CONFIG_PATH=./qvac.config.json npm run cli
```

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

El punto 3 es lo que hace detectable el caso más difícil (INV-1004): ahí los totales **coinciden** —se factura el monto completo por una entrega parcial— y la factura se delata sola, porque sus propios ítems suman 2.100 contra un total de 2.980.

### Manejo de incertidumbre

El prompt de auditoría instruye explícitamente al modelo a responder `UNCERTAIN` cuando no hay evidencia suficiente: si no se recuperó ningún respaldo, si el recuperado es de otro proveedor u otra orden de compra, o si el OCR quedó demasiado corrupto para comparar montos. Inventar una coincidencia es un error mucho más caro que admitir la duda, y el reporte lo dice así.

### Structured outputs deterministas

Los schemas zod de [`src/types.ts`](src/types.ts) son la única fuente de verdad: de ellos se derivan tanto los tipos de TypeScript como los JSON Schema que se le pasan al modelo en `responseFormat: { type: 'json_schema' }`. llama.cpp compila ese schema a una gramática GBNF que restringe la generación token a token, así que la forma del JSON está garantizada a nivel de gramática, no sólo por prompt.

zod es la segunda barrera, porque la gramática garantiza la forma pero no la coherencia. Si la validación falla, se reintenta **una vez** devolviéndole al modelo el error concreto de validación. Si vuelve a fallar, el documento se marca `UNCERTAIN` (o `ERROR`) con el motivo en el reporte — nunca se lanza una excepción sin capturar.

### Manejo de errores

Cada punto de falla degrada a un estado visible en el reporte en vez de tumbar la corrida: modelo que no carga, archivo ilegible, OCR vacío, JSON malformado, PO inexistente, búsqueda RAG fallida. Un documento roto marca esa fila y el lote sigue. Cada etapa se cronometra por separado y las latencias viajan en el resultado.

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
| INV-1004 | PNG | falta un ítem del PO pero se factura el total completo |
| INV-1005 | PDF | sin orden de compra de respaldo → debe dar `UNCERTAIN` |
| INV-1006 | PDF | precio unitario inflado de 12,00 a 13,50 |

Las facturas vienen en PDF y en PNG a propósito, para que cada corrida ejercite tanto la rama de rasterización como la de imagen directa.

## Estructura

```
src/
  types.ts                  schemas zod + JSON Schema para structured outputs
  cli.ts                    entry point del M1
  selftest.ts               verificación offline (npm run verify)
  services/
    qvacService.ts          toda la inferencia @qvac/sdk
    rasterize.ts            PDF → PNG con pdf-to-img
samples/
  generate.ts               generador de datos de prueba
  invoices/  support/       documentos generados
  EXPECTED.md               ground truth
qvac.config.json            config del SDK (loggerConsoleOutput activo)
```

## Hardware

Pensado para un portátil de ~8 GB de RAM. El pico de memoria lo fija la fase 3 (~2,5 GB del LLM más el contexto), porque las fases nunca se solapan. Corre en CPU; si hay GPU disponible el SDK la aprovecha.
