# Best8: estado de restauración y detección de anomalías

## Decisión antes de otra prueba

**No se ejecuta otra continuación.** El binario histórico está en Google Drive,
no en el repositorio local. Solo existen checkpoints smoke de dos épocas en
esta máquina. Por tanto, no se afirma que el best.pt real tenga Adam completo
ni que carezca de él: eso requiere inspeccionar sus bytes después de montar
Drive. Se entrega esa inspección de lectura y un pipeline de anomalías congelado.

La revisión del código sí establece dos hechos:

1. `weights_only` crea Adam nuevo y cambia al prefijo/orden diagnóstico. La
   ventana comunicada no reprodujo las condiciones originales.
2. `continue` en `deterioration_control.py` restaura pesos, Adam, RNG, controles,
   configuración compatible y cursor **de un checkpoint nuevo de Fase 4**.
   Rechaza un best histórico; no restaura el recorrido histórico por épocas.

Las pruebas verifican restauración de la ventana frente a ejecución continua,
pero eso no convierte su muestreo en el del trainer original. No se cambia el rol
de best.pt, no se renombra el archivo para sortear controles y no se mezclan
pesos best8 con Adam/RNG de last30.

## Qué demuestra la ventana comunicada

| Actualizaciones | CE en el mismo prefijo, nats/secuencia |
| ---: | ---: |
| 0 | 160.40 |
| 8 | 227.15 |
| 16 | 193.81 |

El umbral de deterioro sostenido era aproximadamente 1.2·160.40 = **192.48**.
Ambas evaluaciones superan ese umbral; que la segunda mejore respecto a 227.15
no elimina el deterioro frente a la mejor referencia. La parada
`sustained_prefix_reconstruction_deterioration` y la preservación del checkpoint
inicial son coherentes con el diseño del control.

Esto no demuestra que Adam nuevo sea la causa del incidente histórico: también
cambiaron el cursor, la reserva de ejemplos y su orden. No se interpreta como
reproducción del salto de época 12 ni como inutilidad del best8 para inferencia.
La evidencia histórica de reconstrucción específica de época 8 sigue vigente.

## Inspección real de best.pt sin actualizar parámetros

Archivo confirmado:

```text
/content/drive/MyDrive/Workshop_4/runs/workshop4_clipping_v2_prueba12/ae_z32/checkpoints/best.pt
```

`src/checkpoint_fidelity.py` inspecciona ese archivo y, si están disponibles,
metadata.json y last.pt de la misma carpeta. Calcula SHA antes/después, carga con
`torch.load(weights_only=True)` y produce un informe en una carpeta nueva.

Se comprueba:

- Pesos, arquitectura y contrato de vocabulario/splits; finitud de parámetros.
- Correspondencia de IDs de parámetros Adam con el orden del modelo, presencia
  de estados para todos los parámetros, exp_avg, exp_avg_sq y step; formas,
  finitud, segundo momento no negativo y contadores enteros. Se comprueba también
  max_exp_avg_sq cuando el grupo declara AMSGrad.
- LR, betas, eps, weight decay y demás opciones; coherencia entre configuración
  guardada, hparams y metadata. Restaurar grupos Adam puede reemplazar las opciones
  del optimizador recién construido; se compara el estado resultante completo.
- RNG, runtime, clipping FP64, estabilidad y manifest. Se distingue manifest
  presente/completo de bytes de datos efectivamente reverificados.
- Época atribuida a los pesos frente al último registro de history y contadores
  aceptados. Solo se calcula el número esperado de pasos cuando los registros
  completos lo permiten; no se inventa a partir de supuestos sobre lotes omitidos.
- Coincidencia exacta de Adam/RNG con last30 pese a pesos diferentes: sería
  evidencia de mezcla temporal. Ausencia de esa coincidencia no prueba alineación.

La prueba de restauración usa un modelo y Adam desechables. Carga sus state_dict
y compara fingerprints, conservando el RNG/flags del proceso. **No llama
optimizer.step ni hace backward.** Separa explícitamente estas conclusiones:

| Campo | Qué permite concluir |
| --- | --- |
| `adam.structurally_complete` | Existen estados Adam completos y estructuralmente válidos |
| `structural_restore_possible` | Pesos y Adam se restauran mecánicamente sin actualizar |
| `temporal_alignment` | Coherencia, contradicciones o incertidumbre temporal registrada |
| `epoch_boundary_state_restorable` | El envelope de entrenamiento admite una restauración estructural más completa |
| `data_bytes_reverified` | El auditor no lo da por cierto a partir de un hash guardado |
| `historical_faithful_continuation_eligible` | No se habilita automáticamente: falta verificar recorrido, bytes y entorno históricos |

### Por qué puede haber Adam en un best de inferencia

Al guardar un best nuevo durante la época que mejora, el trainer original incluye
el diccionario de Adam/RNG junto con los pesos de esa época. Sin embargo, al
recuperar best desde un last resumible, compone pesos `best_state_dict` y época
best con el envelope de last, que incluye momentos/RNG posteriores. Por eso la
presencia de las claves no basta: hay que revisar qué variante de best existe.
El contrato `best_inference_only` no certifica alineación para reanudar.

Esto es una limitación de procedencia, no evidencia de que ese mecanismo haya
causado el deterioro. El trainer original rechaza best como fuente de resume.

## Condiciones para una única continuación fiel

Solo tendría sentido preparar una continuación ejecutable tras demostrar:

1. Estado completo de pesos/Adam/RNG **de la misma época 8**, sin contradicción
   con history/contadores y con procedencia de guardado confiable.
2. Configuración original completa, incluido batch, LR, opciones Adam, shuffle
   buffer, semilla, transporte, clipping y política numérica.
3. Bytes de train/val iguales al manifest y el iterator original. Desde el final
   de época 8, el siguiente tramo sería el inicio de época 9, con su semilla y
   shuffle originales; no el prefijo permutado de Fase 4.
4. Backend/versiones, GPU, opciones deterministas y estado RNG compatibles. La
   restauración CPU sin pasos no certifica equivalencia bit a bit en T4.

Plan condicional: **una** ventana de hasta 8 actualizaciones, checkpoint inicial
y después de cada update, evaluación sobre los mismos 128 antes/después de cada
paso, guardas numéricas y de deterioro, preservación del mejor, salida nueva y
fuentes intactas. Sin reiniciar Adam ni modificar LR; sin repetir ventanas
automáticamente. El número de época/posición se derivaría del iterator original.

Actualmente esas condiciones no están verificadas sobre el archivo de Drive.
No se entrega una orden que ejecute otra rama weights_only como si fuera fiel.
El notebook primero genera el informe de inspección; no contiene entrenamiento.

## Prioridad paralela: anomalías usando best8 congelado

`src/anomaly_evaluation.py` permite aprovechar la reconstrucción aprendida de
best8 sin depender de que sus momentos Adam sean reanudables. Solo carga los
pesos, llama eval/no_grad, verifica finitud e invariancia del modelo y del SHA
del archivo. **No construye Adam ni restaura el optimizador.**

### Scores y definición de normalidad

El score principal fijado antes de evaluar es:

`MSE_prob = mean_{t,v} (softmax(logits)[t,v] - one_hot(x[t])[v])²`.

Es MSE en la representación categórica one-hot, promediado sobre las 80
posiciones y V categorías. Equivale a Brier multiclase dividido por V. No se
calcula MSE entre IDs de caracteres: la distancia numérica entre categorías
dependería arbitrariamente del vocabulario. NLL de x, sumada sobre posiciones,
se conserva como score auxiliar en nats/secuencia. No se cambia la pérdida con
la que fue entrenado best8 ni se ignora UNK=0.

Sin una etiqueta semántica normal/anormal, se selecciona la clase de siguiente
carácter y más frecuente en **todo train**, con desempate por menor ID. También
se permite fijar una clase explícita con soporte en train. Normal significa
`y_id == normal_class`; los otros y son anomalías proxy. Se registra token,
frecuencias, fingerprint y procedencia. Esa etiqueta no identifica necesariamente
una anomalía lingüística; el rendimiento puede ser bajo y debe medirse.

### Calibración y evaluación sin fuga

- Los hablantes existentes de val se separan por hash/semilla, antes de puntuar,
  en grupos disjuntos de calibración y evaluación de desarrollo.
- Solo ejemplos normales del primer grupo calibran los umbrales. Un reservoir
  por hablante recorre el split completo pero conserva un conjunto acotado;
  evita usar exclusivamente ventanas contiguas de un prefijo y registra cuotas,
  cobertura y recuentos. No se modifican los splits ni datos originales.
- alpha=0.05 se fija de antemano. Se usa el estadístico de orden
  `ceil((n+1)*(1-alpha))`. La alarma es `score > umbral`; los empates quedan
  normales. Si n no permite un umbral finito, se devuelve explícitamente
  `insufficient_normal_calibration`, sin Infinity ni umbral inventado.
- Se guardan thresholds.json **antes** de puntuar evaluación/test. Test está
  desactivado por defecto. `--evaluate-test` habilita su evaluación mixta con
  esos mismos umbrales; jamás se usa test para fijar normalidad, score o umbral.
- Se reportan ROC-AUC y average precision con empates agrupados, matriz de
  confusión, precision, recall, F1, FPR y distribuciones de scores. AUC es
  indefinida si falta una clase; no se sustituye por cero.

El procedimiento de orden no garantiza FPR nominal: los grupos de hablantes
pueden tener distinta distribución y las ventanas pueden solaparse. Los
subconjuntos estratificados no son una estimación sin sesgo de la prevalencia de
todo val/test. Los resultados tampoco equivalen al prefijo de 128 utilizado en
la auditoría de reconstrucción.

### Alcance respecto al objetivo académico

El taller requiere el experimento de anomalías con entrenamiento **normal-only**,
evaluación mixta, MSE por ejemplo y AUC-ROC, con umbral elegido en validación.
El best8 general se entrenó con todas las clases y no cumple por sí solo ese
requisito. El pipeline lo etiqueta como `exploratory_not_normal_only_trained`,
sin presentar métricas exploratorias como el detector final del taller.

Se prepara ahora la infraestructura y la evaluación de best8; quedan pendientes,
sin ejecutarse, el entrenamiento normal-only y el VAE con sus scores de
reconstrucción/KL. La arquitectura permanece igual. Referencia del alcance:
[Workshop_4.pdf](Workshop_4.pdf), apartados de anomalías y consideraciones.

## Ejecución en Colab y entregables

Notebook nuevo: `notebooks/Workshop_4_Best8_Anomalies_Colab.ipynb`.
ZIP nuevo: `exports/workshop4_colab_best8_anomalies.zip`, sin pesos ni Parquet.
Las rutas de best8, last30 y metadata ya están configuradas; la existencia se
comprueba tras montar Drive. Las salidas siempre usan directorios nuevos.

La inspección es de lectura. El notebook permite descargar su JSON para revisar
los campos reales en el repositorio local. Anomalías permanece con
`RUN_ANOMALY_EVALUATION=False`, y test con `EVALUATE_TEST=False`; no hay celda
de entrenamiento ni una matriz automática.

CLI, desde Workshop_4 (primero sin --execute para revisar planes):

```bash
python -m src.checkpoint_fidelity --checkpoint /ruta/best.pt \
  --metadata /ruta/metadata.json --comparison-checkpoint /ruta/last.pt \
  --processed-dir /ruta/processed --output-dir /salida/nueva/inspeccion

python -m src.anomaly_evaluation --checkpoint /ruta/best.pt \
  --processed-dir /ruta/processed --device cuda \
  --output-dir /salida/nueva/anomalias
```

`--execute` en estos módulos habilita **inspección/evaluación**, no entrenamiento.
Anomalías tiene 600 s máximos con supervisión externa, hasta 16.384 ejemplos
puntuados en total y 2.048 por población por defecto. Los recorridos de train
para frecuencias y val para muestreo son de lectura y pueden consumir E/S.
El test opcional añade hasta 2.048 ejemplos por defecto. Un corte duro deja
salidas parciales, que no deben presentarse como evaluación completada.

Salidas de anomalías: plan.json, normal_definition.json, sampling.json,
thresholds.json, scores.csv, metrics.json y summary.json. Incluir el SHA, score
principal y alcance muestral al reportar valores. No se han calculado métricas
de anomalías reales sobre best8 en esta máquina: faltan sus bytes.

Evidencia local y verificaciones: `audit/best8_restore_anomalies/`. Las pruebas
usan fixtures y objetos desechables; no constituyen una inspección del archivo
real ni una ejecución de anomalías de producción. Se conservan los informes,
ZIP, datos y checkpoints anteriores. No se ejecutan entrenamientos completos,
no se cambia arquitectura y no se hace commit ni push.

Validación final: **185 pruebas aprobadas, cero errores y cero omisiones**,
incluidas 36 nuevas (15 de checkpoint/Adam y 21 de anomalías). Los 123 archivos
protegidos conservaron sus hashes. Se verificaron sintaxis/celdas sin resultados,
planes CLI sin efectos y ausencia de comandos de entrenamiento en el notebook.
El contenido, CRC, extracción segura y planes desde el ZIP se verifican en
`package_validation.json`. No se atribuyen valores de Adam o AUC al archivo
histórico que todavía no se ha inspeccionado aquí.
