# AE z32 normal-only: primer entrenamiento y evaluación de anomalías

## Decisión y ejecución preparada

Se avanza a **un entrenamiento nuevo de AE z32 desde cero**, usando todos los
ejemplos normales de train. La arquitectura original permanece: embedding 32,
GRU 128 de una capa, z32 lineal y decoder GRU condicionado por z. No se restaura
best8 ni su Adam. No se vuelve a investigar la causa del deterioro histórico
como condición previa para este experimento. Best8 se conserva como baseline
exploratorio congelado.

El notebook nuevo `notebooks/Workshop_4_NormalOnly_AE_Colab.ipynb` permite una
ejecución controlada en T4, seguida de calibración, métricas y comparación.
`RUN_TRAINING=False` evita iniciarla al abrir el archivo; activar ese control
ejecuta exclusivamente este AE, con salidas nuevas en Drive. AE z64 y VAE
β=0.5, 1 y 4 quedan pendientes; no hay un selector de matriz completa.

## Etiquetas, procedencia y equilibrio de clases

Se mantiene la definición anterior para poder interpretar la comparación:

- **Normal**: y es el siguiente carácter modal de train, escogido sin val/test.
- **Anomalía**: y corresponde a cualquiera de las demás clases; positivo=1.
- El modelo reconstruye **x**, no y. La etiqueta y solo filtra normales y sirve
  para medir detección; no entra al encoder/decoder ni a la pérdida de x.

La clase modal es el espacio, ID1. Es la clase individual más frecuente, aunque
no reúne más del 50% de filas. El resto agregado es mayoritario. El recuento de
los Parquet existentes, sin entrenar ni leer test, dio:

| Split | Total | Normales | Anomalías proxy | Prevalencia de anomalía |
| --- | ---: | ---: | ---: | ---: |
| Train | 2.997.217 | 562.253 | 2.434.964 | 81.2408% |
| Val completo | 583.098 | 109.197 | 473.901 | 81.2730% |

Los conteos y SHA de fuentes están en `audit/normal_only_ae/class_inventory.json`.
Normalidad procede exclusivamente de train y se vuelve a registrar en la
ejecución. No se cambia la definición después de ver AUC, ni se invierten scores
para favorecer una métrica. Son **etiquetas proxy de siguiente carácter**:
no equivalen a rareza lingüística, fraude o novedad semántica. Normal-only puede
mejorar la tarea definida, pero no se garantiza que reconstrucción detecte bien
estas clases que comparten mucho contenido de x.

### Interpretación del baseline comunicado

El usuario comunicó estos resultados exploratorios sobre 2.021 ejemplos:

| Score | ROC-AUC | Average Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| MSE categórica | 0.490708 | 0.817690 | 0.06424 | 0.11917 |
| NLL | 0.509244 | 0.826352 | 0.08000 | 0.14634 |

ROC-AUC está cerca de 0.5: no hay discriminación útil demostrada. AP debe
compararse con la prevalencia de anomalías **de esa misma muestra**. Un score
constante obtiene AP igual a dicha prevalencia; AP≈0.82 no demuestra una buena
clasificación en un conjunto con aproximadamente81% de positivos. No se infiere
el balance exacto de los 2.021 ejemplos a partir de AP ni se inventa su matriz
de confusión. Faltan sus filas/contadores para esa comparación exacta.

Estos valores son AP, **no PR-AUC trapezoidal**. El nuevo evaluador reporta ambas
por separado y define la interpolación, especialmente relevante con empates.

## Separación de entrenamiento, selección, calibración y evaluación

Los splits originales ya están separados por hablante. Dentro de val se fijan,
por hash de hablante y seed 42, tres roles disjuntos antes de calcular scores:

| Rol val | Hablantes | Filas originales | Normales originales | Uso permitido |
| --- | ---: | ---: | ---: | --- |
| model_selection | 57 | 152.927 | 28.797 | Seleccionar/early-stop usando solo pérdida de normales |
| calibration | 56 | 204.635 | 38.147 | Fijar umbrales usando solo normales |
| evaluation | 56 | 225.536 | 42.253 | Medir detección mixta; no ajustar modelo ni umbral |

Separar solo calibración y evaluación era insuficiente si alguno de esos mismos
ejemplos se utilizaba para early stopping. La selección ahora tiene un tercer
grupo exclusivo. Test se mantiene fuera de entrenamiento, elección de modelo,
score y calibración. Sus hablantes solo se toman de metadata hasta activar la
etapa final; no se leen sus ejemplos durante preparación/entrenamiento.

### Coincidencias exactas encontradas y corrección

La revisión de x por SHA256, independiente de y e ID de hablante, encontró:

- 4 normales de model_selection idénticos a algún x de train normal.
- 60 entradas del evaluation mixto idénticas a algún x de train normal.
- Ninguna coincidencia de ese tipo en calibration normal, ni entre los tres
  conjuntos de validación examinados entre sí.

El pipeline construye una guarda con todos los x de train normal y excluye
coincidencias de las **vistas held-out antes del reservoir y de cualquier
forward**. No cambia Parquet ni elimina normales de entrenamiento: conserva las
562.253 filas, incluidas sus repeticiones internas. Guarda digest, conteos y
cuarentenas. La evaluación vuelve a derivar y verificar la guarda de train antes
de puntuar; test no permite sustituir datos entrenados ni recalibrar.

Además, best8 vio todas las clases de train. Una comprobación sobre las filas
limpias seleccionadas encontró20 hashes de evaluation presentes en train
no-normal. Por ello, **cuando se incluye best8**, una segunda cuarentena compara
las filas retenidas contra todo train y retira coincidencias de AMBOS modelos,
manteniendo la comparación pareada. Se aplica también en la etapa final de
test, sin releer calibration. No se modifica best8 ni su dataset histórico.

Se registran filas/inputs excluidos y fingerprints de las poblaciones finales.
No se afirma deduplicación semántica o de paráfrasis: la comprobación es exacta.
La separación de hablantes tampoco vuelve independientes las ventanas solapadas;
no se anuncian intervalos IID ni FPR nominal garantizada.

Una preparación local sin scores, con seed 42 y cuotas máximas 4096/4096/8192,
seleccionó3.433 normales de model_selection,3.353 de calibration y7.660 ejemplos
mixtos de evaluation tras la primera cuarentena. Esta última muestra contenía
6.269 anomalías (81.8407%). La cuarentena adicional para best8 puede reducirla;
los resultados deben usar los conteos finales de `sampling`/métricas, no asumir
que el máximo solicitado se alcanza. Evidencia: `clean_protocol_preview.json`.

## Entrenamiento normal-only de principio a fin

Configuración en `config/normal_only_ae.json`:

| Control | Configuración inicial |
| --- | --- |
| Arquitectura | AE original, embedding 32/hidden 128/z32/una capa/dropout0 |
| Inicialización | Pesos nuevos y Adam nuevo, seed 42; sin fuente histórica |
| Datos | Todos los normales de train, filtrados por y antes de formar batches |
| Batch y mezcla | 128; buffer local acotado de 16 batches; shuffle seed+época |
| Optimización | Adam lr=0.0003, defaults originales; CE de x |
| Presupuesto | Hasta30 épocas,150.000 updates y4 h; techo absoluto6 h |
| Validación de selección | Normales de model_selection, hasta 4.096 por cuotas |
| Evaluación/checkpoint | Cada512 updates y final de época; checkpoint inicial |
| Early stopping | 5 evaluaciones sin mejora relevante de 0.1 nat/secuencia |
| Warmup | 3 épocas normales completas antes de aplicar early stopping por plateau |
| Latente | Swap de códigos sin puntos fijos y TV sobre un subconjunto fijo≤128 normales |

Con los datos actuales, una época normal son562.253 ejemplos y4.393 batches,
incluido el último parcial. Treinta épocas serían131.790 updates si no hay
paradas. No se usa un prefijo fijo de 12.288 ejemplos ni se cambian datos a mitad
del entrenamiento. La pérdida pesa correctamente los lotes parciales.

El LR es una elección conservadora para el primer experimento desde cero;
no se presenta como reparación demostrada ni ganador de una búsqueda. La
arquitectura y CE permanecen intactas. No se añade regularización o teacher
forcing para obtener ventaja aparente.

Se registran pérdida train media de la época y media acumulada diferenciadas,
pérdida normal de selección, daño reconstructivo por swap, TV, magnitud z,
fracción de clipping y normas pre/post. Un swap grande puede ser dependencia
arbitraria: se interpreta junto con reconstrucción. Un swap inicial pequeño no
detiene un modelo que aún no ha aprendido.

### Guardas y mejor checkpoint

Se importa el clipping seguro existente: norma y escalado FP64, finitud previa
y postcondición≤5·(1+1e−6). Se verifica pérdida/z/gradientes antes de Adam y
pesos/momentos después de cada paso. No se permite Adam con gradientes inválidos
ni se omiten lotes silenciosamente. Se comprueba reset de gradientes por batch.

Paradas: NaN/Inf o norma inválida; norma raw>1e8; magnitud z>100; CE normal de
selección≥2×su mejor referencia sana; CE≥1.2×durante2 evaluaciones; o retención
de utilidad swap≤50% durante2 evaluaciones después de una referencia previa
informativa≥1 nat/secuencia. Clipping100% por sí solo no fuerza una parada.
Estas guardas permanecen activas durante warmup; solo se retrasa el plateau.

Los checkpoints periódicos son inmutables y se confirman mediante temporal y
renombrado. El mejor puntero solo promueve menor CE en un estado sano; un
checkpoint deteriorado no lo reemplaza. El modelo inicial se conserva como
protección, pero no se exporta como detector entrenado si sigue siendo el mejor.
`best_inference.pt` se exporta solo desde un candidato con actualizaciones
aceptadas. Los roles inicial/smoke/periódico no son candidatos finales válidos.

Los estados periódicos conservan Adam/RNG/configuración para inspección, pero
este primer runner **no implementa resume** ni afirma recuperar un buffer
interrumpido a mitad de época. Un corte preserva checkpoints confirmados; no
inicia otra ejecución. Reportar tiempo/épocas/updates realmente completados.

## Calibración y evaluación congeladas

El score principal predeclarado es `brier_mse`:

`mean_{t,v}(softmax(logits)[t,v] - one_hot(x[t])[v])²`.

Es MSE categórica, no distancia ordinal entre IDs. NLL es la suma de CE de x
sobre las 80 posiciones, en nats/secuencia. Para ambos, más alto significa más
anómalo; anomalía es el positivo. No se invierte un score por observar AUC<0.5.

Cada score/modelo calibra exclusivamente con normales de calibration, usando
alpha=.05 y el estadístico de orden `ceil((n+1)*(1-alpha))`. `score>umbral` genera
alarma y los empates quedan normales. Si la muestra no permite un umbral finito,
se declara insuficiente y no se habilita test; no se fabrica Infinity como umbral.

Antes del primer forward mixto se sella `frozen_protocol.json` y su SHA con:
modelos y hashes, clase normal, rol/digests, scores/dirección, umbrales y filas
retenidas. La evaluación posterior no selecciona otro checkpoint ni retoca el
umbral. Test es una etapa aparte que exige el protocolo sellado y verifica que
configuración, datos/guardas y modelos coincidan. No recalibra ni relee val.

Se reportan por modelo y score:

- ROC-AUC; referencia de azar0.5 cuando existen ambas clases.
- Average Precision por grupos de scores empatados y prevalencia positiva.
- PR-AUC trapezoidal, con convención inicial recall0/precision1 documentada.
  Con empates puede diferir mucho de AP: un score constante tiene AP igual a
  prevalencia, mientras el área trapezoidal puede ser mayor por interpolación.
- Precision, recall, F1, FPR y matriz TP/FP/TN/FN en el umbral congelado.
- Curvas ROC/PR, distribuciones normal/anómala, tamaños y cobertura de hablantes.

Si falta una clase, ROC-AUC queda indefinida. Una muestra estratificada por
hablante no reproduce exactamente la prevalencia completa; se usa la prevalencia
observada para interpretar AP. El umbral de 5% no garantiza FPR real5% con shift
de hablantes o dependencia de ventanas.

## Comparación con best8 y alcance de los resultados

La comparación formal vuelve a evaluar el best8 histórico congelado y el mejor
normal-only sobre **las mismas filas actuales**, después de las cuarentenas,
con umbrales propios calibrados sobre las mismas entradas normales. No restaura
Adam de best8 ni lo usa para inicializar el nuevo AE.

Hay diferencias metodológicas: best8 aprendió con todas las clases y fue elegido
con la validación histórica completa; el nuevo modelo aprende solo normales y
se elige con model_selection separado. El presupuesto y la población de
entrenamiento también difieren. Los números comunicados sobre 2.021 ejemplos
son contexto no pareado mientras no coincidan fingerprints; no se restan como
si fueran el control de la nueva muestra.

Los resultados mixtos de development orientan la interpretación. Test se abre
solo una vez fijado el protocolo; no se escogen score, clase normal o umbral
después de verlo. No se anuncian mejoras de detección hasta ejecutar el primer
entrenamiento real y evaluar sus scores. Las métricas actuales de best8 cercanas
a azar tampoco garantizan que el normal-only vaya a resolver la etiqueta proxy.

## Entregables y ejecución

- `src/normal_only_protocol.py`: roles, filtro normal completo, hashes/cuarentenas.
- `src/train_normal_ae.py`: un solo entrenamiento AE z32 desde cero.
- `src/evaluate_normal_ae.py`: calibración, comparación y test sellado.
- `notebooks/Workshop_4_NormalOnly_AE_Colab.ipynb`: flujo T4 con salidas Drive.
- `exports/workshop4_colab_normal_only_ae.zip`: paquete nuevo, sin datos ni pesos.

Desde Workshop_4, los CLI muestran planes sin efectos salvo añadir `--execute`:

```bash
python -m src.train_normal_ae --config config/normal_only_ae.json \
  --processed-dir /ruta/processed --device cuda --output-dir /drive/nuevo/train

python -m src.evaluate_normal_ae --stage evaluate \
  --checkpoint /drive/nuevo/train/best_inference.pt \
  --baseline-checkpoint /drive/historico/best.pt \
  --processed-dir /ruta/processed --device cuda --output-dir /drive/nuevo/dev

python -m src.evaluate_normal_ae --stage test \
  --checkpoint /drive/nuevo/train/best_inference.pt \
  --frozen-protocol /drive/nuevo/dev/frozen_protocol.json \
  --processed-dir /ruta/processed --device cuda --output-dir /drive/nuevo/test
```

El notebook copia únicamente train/val a Colab local para entrenar, conserva
outputs/checkpoints en una carpeta nueva de Drive y deja test en su origen hasta
activar la etapa final. Al reabrir Colab puede indicarse la carpeta entrenada
para evaluar sin volver a entrenar. No se borran ni sobrescriben modelos previos.

La validación local usa fixtures y entrenamiento sintético mínimo; no ejecuta
las 30 épocas de Shakespeare en CPU ni produce métricas nuevas de detección.
La evidencia de datos y los resultados de pruebas/paquete se guardan en
`audit/normal_only_ae/`. No se modifica la arquitectura ni el clipping original;
no se hace commit ni push. El siguiente experimento, tras estos resultados,
podrá abordar AE z64 y los VAE pendientes con el mismo aislamiento de datos.

Validación final local: **235 pruebas aprobadas, cero errores y cero omisiones**,
incluidas 50 nuevas (17 de protocolo/datos, 13 del entrenamiento y 20 de
evaluación). Los 141 archivos protegidos de entregas anteriores conservaron
sus hashes. Se comprobaron las celdas del notebook sin ejecutar, los planes CLI
sin efectos y el paquete/extracción. No se ejecutó entrenamiento de Shakespeare
ni se calcularon métricas nuevas del detector normal-only en esta máquina.
