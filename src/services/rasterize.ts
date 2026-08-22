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
 * `scale: 2` duplica la resolución respecto del tamaño natural de la página;
 * a 1x el texto de cuerpo de una A4 queda demasiado chico y el reconocedor
 * empieza a confundir dígitos, que es justo lo que no se puede permitir en
 * montos de facturas.
 */
export async function toImagePages(filePath: string): Promise<string[]> {
  const ext = path.extname(filePath).toLowerCase()

  if (ext !== '.pdf') return [filePath]

  const { pdf } = await import('pdf-to-img')
  const document = await pdf(filePath, { scale: 2 })
  const outDir = await mkdtemp(path.join(tmpdir(), 'qvac-raster-'))
  const base = path.basename(filePath, ext)

  const pages: string[] = []
  let pageNumber = 1
  for await (const page of document) {
    const out = path.join(outDir, `${base}-p${pageNumber}.png`)
    await writeFile(out, page)
    pages.push(out)
    pageNumber++
  }
  await document.destroy()

  if (pages.length === 0) {
    throw new Error(`El PDF no produjo ninguna página rasterizable: ${filePath}`)
  }
  return pages
}
