# Clipping global estable y recuperación numérica

## Diagnóstico y límites de lo conocido

Los diagnósticos proporcionados desde Colab para la época 12, lotes 22400 y
22651, muestran gradientes con nan=0 e inf=0 y normas FP64 de aproximadamente
2.836e19 y 8.164e20. Es un desbordamiento en la reducción FP32 del clipping.
Esas dos normas todavía caben como números FP32, pero sus cuadrados superan el
rango FP32: sqrt(max(FP32)) es aproximadamente 1.845e19. También se prueba una
norma global mayor que max(FP32), con elementos individuales todavía finitos.

En el AE revisado, el objetivo es x, las dimensiones son correctas y la pérdida
usa entropía cruzada sobre logits. No se encontró un error algebraico evidente
en el backward, el decoder GRU o la pérdida. El AE no calcula KL ni exp(logvar).
La suma de 80 posiciones y la recurrencia del encoder/decoder pueden amplificar
gradientes. Corregir su reducción numérica no elimina esa posible inestabilidad
del modelo: las magnitudes grandes se siguen registrando para su análisis.
No se inspeccionó ni modificó el checkpoint real de época 11 en Drive durante
esta corrección; la información de los dos incidentes procede del usuario.

No se cambian arquitectura, learning rate, batch, Adam, pérdidas ni particiones.
El umbral sigue siendo exactamente **grad_clip=5.0**. Tampoco se usa nan_to_num
ni error_if_nonfinite=False. La verificación explícita del nuevo clipping
sustituye la comprobación de norma del operador nativo FP32.

## Cálculo y escalado estables

`stable_clip_grad_norm_` implementa clipping L2 global, conjuntamente sobre
todos los parámetros:

1. Verifica que todos los elementos de todos los gradientes sean finitos,
   antes de modificar cualquiera de ellos. NaN/Inf causa rechazo.
2. Convierte cada gradiente a FP64 **antes** de calcular las normas; combina
   las normas parciales también en FP64.
3. Calcula un único factor `min(1, 5.0 / norma_global)`; para norma cero usa 1.
4. Multiplica cada gradiente en FP64 y después copia el resultado a su dtype
   original. No convierte primero el coeficiente a FP32: con normas muy grandes
   ese coeficiente puede ser subnormal y perder precisión o subdesbordar.
5. Comprueba de nuevo los elementos y la norma FP64 de los gradientes ya
   convertidos, que son los que recibirá Adam. Exige norma finita y no mayor
   que 5.0 dentro de una tolerancia relativa de 1e-6 por redondeo.

Se conserva la dirección global y los signos del gradiente. Los gradientes con
norma menor o igual al umbral no se modifican. La implementación admite
gradientes densos FP32/FP64 en un solo dispositivo, como el entrenador actual;
no habilita AMP ni introduce cambios de precisión en forward/backward.
Si falla la condición posterior o aparece un error de CUDA/clipping ajeno a
la comprobación numérica, se aborta antes de optimizer.step.

Los gradientes finitos grandes ahora se escalan y aceptan después de comprobar
el resultado. **No consumen el presupuesto de lotes omitidos.** Las magnitudes
originales siguen siendo evidencia relevante para investigar gradientes
explosivos: aceptar el clipping no demuestra que el modelo esté bien ajustado.

## Registros del clipping

- `gradient_norms.csv`: una fila por lote con clipping verificado, con sesión,
  época, índice de lote, norma previa/posterior, factor y marcas de clipping y
  riesgo de desbordamiento FP32. Se escribe en modo append, con buffering y
  flush periódico, al cerrar la época y ante eventos relevantes.
- `nonfinite_events.jsonl`: los gradientes finitos con norma superior a
  sqrt(max(FP32)) generan `large_finite_gradient_clipped`, con normas, factor,
  umbral y acción `accepted_after_verified_clipping`. Los NaN/Inf conservan los
  diagnósticos de rechazo descritos abajo.
- `history.csv`: añade media/máximo de norma previa, máximo posterior y número
  de lotes clippeados y con gradientes grandes, con prefijo train_.
- Metadatos/checkpoints: registran `gradient_clipping.algorithm=global_l2_fp64_v1`
  y la tolerancia fuera de training_config. Las épocas históricas mantienen
  sus métricas originales y dejan vacíos los nuevos campos no registrados.

Un registro de clipping describe un paso verificado, no una época completada.
Si se aborta o se repite una época, la sesión permite distinguir los intentos;
los checkpoints siguen guardándose solo en límites de época completos.

## Rechazo explícito y auditable

Un rechazo numérico del clipping se registra en `nonfinite_events.jsonl` con:

- Época, índice de lote desde 1, sesión, dispositivo, forma, rango y SHA-256
  de los IDs del lote. La semilla y época permiten reproducir el orden.
- Nombre, forma y dtype de los parámetros con gradientes; conteos NaN/Inf,
  primeros índices inválidos, máximo absoluto finito y normas diagnósticas FP64.
- Motivo `nonfinite_gradient_elements` (NaN/Inf) o `nonfinite_gradient_norm`
  (norma no representable incluso en FP64), error
  original, acción skip/abort, límites y estado acumulado de incidencias.

**No se ejecuta optimizer.step para el lote rechazado.** Se limpian sus
gradientes con zero_grad(set_to_none=True), también al abortar. Adam no recibe
esos gradientes. Se continúa solo si el modelo/Adam siguen finitos y se respeta
la siguiente política conservadora, registrada fuera de training_config:

| Límite por defecto | Valor |
| --- | ---: |
| Lotes rechazados por época | 1 |
| Lotes rechazados acumulados en la continuación | 2 |
| Lotes rechazados consecutivos | 1 |
| Fracción de ejemplos rechazados por época | 0.001 (0,1 %) |

Superar cualquier límite aborta. La fracción se comprueba al cerrar la época;
si se supera, la época no se publica como checkpoint. Dos rechazos consecutivos
o en una misma época detienen la ejecución inmediatamente. Los límites enteros
pueden ser 0 para detenerse ante el primer incidente. Las opciones CLI son
`--max-skipped-batches-per-epoch`, `--max-skipped-batches-total`,
`--max-consecutive-skips` y `--max-skipped-fraction`. No ampliarlos solo para
conseguir completar el entrenamiento: cualquier cambio exige una justificación.

El contador acumulado se guarda también en `stability_state.json`, incluso
si aborta una época, para que reanudar el mismo directorio no reinicie el
presupuesto de incidencias. El journal se conserva en modo append. La política
y el estado se incluyen en nuevos checkpoints; los v1 anteriores sin esos
campos siguen siendo compatibles. No se agregan claves a su training_config.

Los promedios train usan **solo ejemplos aceptados** y conservan su definición
de pérdida. El CSV añade train_attempted_batches, train_accepted_batches,
train_skipped_batches, train_attempted_examples, train_skipped_examples y
train_skipped_fraction. Los conteos/advertencias evitan interpretar una época
con exclusiones como si hubiera utilizado todos los datos. Las filas antiguas
mantienen estas columnas vacías: no se inventan conteos retrospectivos.
Validación nunca omite ejemplos. Pérdida no finita, excepción de backward o
estado no finito de modelo/Adam abortan, con diagnóstico; no son recuperaciones
silenciosas. La época abortada no reemplaza el último checkpoint completo.

## Continuar desde last.pt de época 11 sin tocar el original

1. Regenerar/subir `exports/workshop4_colab.zip` y extraerlo en una carpeta
   **nueva** de /content (el helper rechaza mezclar versiones). Mantener los
   mismos Parquet, vocabulario y splits. No volver a descargar Shakespeare.
2. Mantener el directorio original de Drive intacto. Usar como fuente su
   `ae_z32/checkpoints/last.pt` y como salida una carpeta nueva, por ejemplo
   `runs/workshop4_fp64`. Conservar también el intento fallido y sus logs.
   No copiar encima de last.pt ni best.pt originales.
3. Desde el Workshop_4 extraído, ejecutar en Colab una llamada como esta,
   sustituyendo las rutas por las reales:

```python
import subprocess, sys
command = [
    sys.executable, "-m", "src.train_ae",
    "--experiment", "ae_z32",
    "--processed-dir", "/content/workshop4_data/processed",
    "--device", "cuda",
    "--epochs", "30", "--batch-size", "128",
    "--learning-rate", "0.001", "--seed", "42",
    "--resume", "/content/drive/MyDrive/Workshop_4/runs/workshop4/ae_z32/checkpoints/last.pt",
    "--output-dir", "/content/drive/MyDrive/Workshop_4/runs/workshop4_fp64",
    "--stop-after-epoch", "12",
]
subprocess.run(command, cwd=WORKSHOP_DIR, check=True)
```

El ejemplo hace únicamente la siguiente época completa para comprobar el
estado en la T4; **no fue ejecutado durante esta corrección**. Si el checkpoint
original contiene otros valores de hidden/embedding/dropout/etc., conservarlos
exactamente mediante YAML/CLI. El entrenador rechaza cualquier diferencia de
configuración o bytes del dataset; no debe eludirse esa comprobación editando
el checkpoint. No usar --smoke-test al reanudar un entrenamiento real porque
cambiaría su training_config.

Se conserva el objetivo de 30 épocas y se restaura modelo, Adam, RNG e historial
de last.pt. Con epoch=11 empieza en 12; best_state/best_epoch se recuperan sin
reiniciarlos (8 si ese es el valor del original). Se valida que los tensores
cargados, incluido Adam y el mejor estado, sean finitos: que el archivo sea
legible no basta. La fuente se abre en lectura y su SHA-256 queda en metadata.
La continuación crea sus propios last.pt/best.pt antes de entrenar, de modo que
otra desconexión se pueda recuperar desde la carpeta nueva.
El origen inicial y su hash se conservan también como continuation_origin en
los siguientes checkpoints y metadatos, aunque después se utilice --resume auto.

4. Revisar el journal, los conteos del CSV y las pérdidas de validación. Para
   continuar esa carpeta, usar `--resume auto` con la **misma** configuración y
   salida, quitando --stop-after-epoch. Esto actualiza los checkpoints de la
   continuación; los originales de la época 11 permanecen sin cambios.

Para una investigación puntual se puede añadir `--detect-anomaly`. Es lento y
detiene el backward ante NaN con traza del forward; no debe confundirse con la
política normal de rechazo. Los nuevos campos de seguridad son independientes
de lr/batch/arquitectura; deben mantenerse iguales al reanudar la continuación.

## Validación local

Resultado de esta corrección: **22 pruebas aprobadas en CPU, sin errores ni
omisiones** (7 de clipping FP64, 9 de estabilidad/integración y 6 de entrenamiento
y recuperación). No se ejecutó entrenamiento completo ni se accedió a Drive.

Se usan tests mínimos con las normas reportadas desde Colab, normas globales
superiores a max(FP32), un coeficiente FP32 subnormal, preservación de dirección,
clipping global entre parámetros y comparación con el operador nativo en el
rango ordinario. Se comprueba siempre la norma posterior real en FP64.

La integración inyecta gradientes NaN/Inf mediante hooks y gradientes finitos
de 1e20. Verifica rechazo de los inválidos, aceptación segura de los finitos
grandes, estado de Adam, logs, promedios, límites, limpieza y persistencia del
presupuesto. Se conserva la prueba de carga de checkpoint v1 sin campos nuevos
en otra salida, comprobando los hashes originales. Estas pruebas locales en CPU
no equivalen a repetir el entrenamiento ni a evaluar su calidad en la T4.
