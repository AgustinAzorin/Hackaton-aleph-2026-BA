# Ground truth de los datos de prueba

Generado por `samples/generate.ts`. El pipeline debería reproducir estos veredictos.

| Factura | PO esperado | Veredicto esperado | Motivo |
| --- | --- | --- | --- |
| INV-1001 | PO-5001 | `MATCH` | Ítems, cantidades y total idénticos. |
| INV-1002 | PO-5002 | `MATCH` | Ítems, cantidades y total idénticos (factura escaneada a PNG). |
| INV-1003 | PO-5003 | `DISCREPANCY` | Recargo de combustible de 420.00 no autorizado en el PO (4,620.00 vs 4,200.00). |
| INV-1004 | PO-5004 | `DISCREPANCY` | Falta el ítem "Cordless drill 18V" pero se factura el total completo de 2,980.00. |
| INV-1005 | — | `UNCERTAIN` | No existe orden de compra de respaldo; no hay evidencia para validar. |
| INV-1006 | PO-5006 | `DISCREPANCY` | Precio unitario de "Archive box" inflado 12.00 → 13.50 (915.00 vs 840.00). |
