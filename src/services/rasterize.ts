/**
 * Rasterización de PDF a PNG.
 *
 * El motor OCR de QVAC opera sobre imágenes, así que todo PDF debe convertirse
 * a PNG (una imagen por página) antes de pasar por OCR. Las imágenes ya
 * soportadas (PNG/JPG) se dejan pasar sin tocar.
 */
import { mkdtemp, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { PDF_RASTER_SCALE } from './tuning.js'

/** Extensiones que el pipeline acepta como documento de entrada. */
export const SUPPORTED_EXTENSIONS = ['.pdf', '.png', '.jpg', '.jpeg'] as const

export function isSupportedDocument(filePath: string): boolean {
  const ext = path.extname(filePath).toLowerCase()
  return (SUPPORTED_EXTENSIONS as readonly string[]).includes(ext)
}

/**
 * Devuelve las rutas de imagen listas para OCR de un documento.
 *
 * - PDF  → se rasteriza cada página a PNG en un directorio temporal.
 * - PNG/JPG → se devuelve la ruta original.
 *
 * `PDF_RASTER_SCALE` duplica la resolución respecto del tamaño natural de la
 * página; a 1x el texto de cuerpo de una A4 queda demasiado chico y el
 * reconocedor empieza a confundir dígitos, que es justo lo que no se puede
 * permitir en montos de facturas. El factor vive en `tuning.ts` porque forma
 * pareja con `OCR_MAG_RATIO`: la resolución que ve el detector es el producto
 * de los dos, y sólo tiene sentido razonarlos juntos.
 *
 * Las páginas se escriben a disco en paralelo: son E/S pura y el OCR no puede
 * arrancar hasta que esté la primera, así que serializarlas era latencia
 * regalada.
 */
export async function toImagePages(filePath: string): Promise<string[]> {
  const ext = path.extname(filePath).toLowerCase()

  if (ext !== '.pdf') return [filePath]

  const { pdf } = await import('pdf-to-img')
  const document = await pdf(filePath, { scale: PDF_RASTER_SCALE })
  const outDir = await mkdtemp(path.join(tmpdir(), 'qvac-raster-'))
  const base = path.basename(filePath, ext)

  const pages: string[] = []
  const writes: Promise<void>[] = []
  let pageNumber = 1
  for await (const page of document) {
    const out = path.join(outDir, `${base}-p${pageNumber}.png`)
    writes.push(writeFile(out, page))
    pages.push(out)
    pageNumber++
  }
  await Promise.all(writes)
  await document.destroy()

  if (pages.length === 0) {
    throw new Error(`El PDF no produjo ninguna página rasterizable: ${filePath}`)
  }
  return pages
}
