# Auditoría de estabilidad y rendimiento — Workshop 4

## Alcance y evidencia disponible

La auditoría modifica únicamente archivos locales de Workshop_4. No se conectó
con Colab, no se intervino el proceso activo ae_z32 de época 26 y no se modificó
ningún archivo de Drive. No se ejecutaron experimentos completos.

Se revisaron autoencoder.py, train_ae.py, data.py, configuración, controlador
Colab y pruebas. El inventario local solo contiene los checkpoints y CSV de
`runs/workshop4/_smoke_real/{smoke_ae,smoke_vae}`: son pruebas de dos épocas,
no best8/last26 de producción. Se guardaron sus hashes antes de trabajar.
No existen localmente copias de los gradient_norms/nonfinite_events de Colab.

Los valores de la siguiente tabla proceden **de la solicitud del usuario**,
no de una inspección de Drive. Su transcripción y cálculos derivados están en
`audit/evidence/user_reported_history.json`. No son nuevas evaluaciones.

## Evolución de las pérdidas hasta época 26

| Época | Train, nats/secuencia | Val, nats/secuencia |
| ---: | ---: | ---: |
| 1 | 219.17 | 209.09 |
| 4 | 178.63 | 184.66 |
| 8 | 164.70 | 168.67 |
| 11 | 159.07 | 172.69 |
| 12 | 709.80 | 1212.92 |
| 13 | 1043.41 | 336.01 |
| 14 | 882.82 | 1131.97 |
| 15 | 1490.27 | 1333.13 |
| 20 | 979.90 | 710.68 |
| 22 | 258.62 | 249.79 |
| 24 | 251.65 | 249.33 |
| 25 | 249.47 | 249.01 |
| 26 | 249.20 | 250.66 |

- Hasta época 8 hay aprendizaje útil. Entre 8 y 11 train mejora, mientras val
  empeora aproximadamente 2,4 %; esto puede indicar sobreajuste o variabilidad,
  pero no establece una causa de la crisis posterior.
- Desde época 12 empeoran **train y val** de forma abrupta y oscilante. El
  sobreajuste habitual no explica por sí solo ese patrón. Es consistente con
  deterioro de optimización/estado del modelo, no con un simple aumento del
  tiempo de lectura de datos.
- La recuperación parcial de épocas 22–26 sigue lejos del mejor modelo:
  val26/best8 = 1,486, aproximadamente 48,6 % más pérdida. Dividiendo por 80,
  val8=2,1084 y val26=3,1333 nats/carácter; sus perplexities derivadas son
  aproximadamente 8,24 y 22,95.
- La CE uniforme de 76 categorías sería 80*ln(76)=346,46 nats/secuencia. La
  meseta cerca de 249 mejora ese valor, pero no demuestra reconstrucción útil
  de cada entrada. Debe compararse contra un predictor marginal entrenado solo
  con train y comprobar dependencia de z. No se ha medido ese baseline aquí.
- Las épocas con pérdidas muy superiores a 346,46 implican predicciones
  penalizadas más que la distribución uniforme en promedio. Es compatible con
  predicciones equivocadas muy confiadas; faltan logits reales para confirmarlo.

## Hallazgos confirmados y causas aún no demostradas

| Estado | Hallazgo | Implicación |
| --- | --- | --- |
| Demostrado en código/tests | CE recibe logits `(B,80,76)` y objetivos x `(B,80)`; suma posiciones y promedia ejemplos | No se encontró un broadcast, softmax duplicado o uso erróneo de y |
| Demostrado | UNK=0, sin PAD; las 80 posiciones se reconstruyen | No hay pérdida de posiciones por máscara/ignore_index=0 |
| Demostrado | Decoder usa una llamada nn.GRU sobre toda la secuencia | No existe un bucle Python de 80 llamadas que vectorizar |
| Demostrado | Gradientes finitos pueden desbordar una reducción FP32 | El clipping FP64 corrige esa operación, conservando el umbral 5 |
| Reportado por usuario | Dos normas de 2.836e19 y 8.164e20, con nan=inf=0 | Evidencia de gradientes enormes y overflow de la reducción anterior |
| Reportado + aritmética | ceil(2.997.217/128)=23.416 lotes, todos recortados en época 25 | Fracción de clipping=100 % en esa época |
| Demostrado en código | z es proyección lineal sin cota; se repite como entrada en los 80 pasos | Riesgo estructural; tanh solo acota el estado inicial del decoder |
| Hipótesis | Jacobianos recurrentes mal condicionados, magnitud de z, saturación de gates o parámetros deteriorados | Necesita comparar best8/last y probes por bloque |
| Hipótesis | Pérdida de dependencia de la entrada o predictor casi marginal en la meseta | Necesita comparar reconstrucción con z original/permutado/cero |
| No demostrado | Que el clipping FP64 causara el deterioro | No hay una reproducción controlada con los mismos pesos/lotes/backend |

**La causa primaria de la degradación del modelo de producción no está
establecida.** Sería incorrecto anunciar que se reparó sin examinar sus pesos
y trazas. Se corrigieron/optimizaron componentes verificables y se prepararon
pruebas que permiten localizar la causa sin otro entrenamiento largo.

### GRU, latente, inicialización y Adam

El encoder GRU usa hidden[-1]; su estado está acotado por las operaciones GRU
con h0=0, pero eso no garantiza derivadas pequeñas respecto a parámetros. Las
matrices recurrentes entrenadas y productos de Jacobianos a lo largo de 80 pasos
pueden amplificar el backward. `to_latent` no limita z. `tanh(latent_to_hidden(z))`
limita h0, pero no la otra ruta de z hacia todos los inputs del decoder.

Se conserva la inicialización por defecto de PyTorch; no hay inicialización
ortogonal explícita. La [documentación GRU 2.14](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html)
describe ecuaciones e inicialización uniforme. Sus pesos por gate y sus normas
son indicadores; una norma singular de weight_hh mayor que 1 **no es** una
prueba de que el Jacobiano efectivo recurrente tenga norma mayor que 1.

La derivada de CE respecto a cada logit es `(softmax-one_hot)/B` bajo esta
reducción; está acotada en magnitud por 1/B. La suma de 80 contribuciones puede
aumentar la escala de los parámetros, pero no explica por sí sola 1e19–1e20.
El AE no calcula KL ni exp(logvar); esos términos no explican ae_z32.

Clipping global limita el gradiente entregado a Adam, no la magnitud de todos
los cambios de pesos: Adam usa momentos y denominadores por coordenada. Si un
bloque domina el gradiente, reduce también bloques sanos. Por ejemplo, para
norma 7,89e15 el factor es aproximadamente 6,34e-16. No debe sustituirse la media
de coeficientes por 5/media(normas). No se cambió el clipping a uno por parámetro.

El post-clipping reportado 5,0000000677 supera 5 solo en aproximadamente
1,35e-8 relativo, dentro de la tolerancia 1e-6; no constituye evidencia de un
fallo del umbral. El 100 % de clipping, aisladamente, tampoco demuestra un bug:
la CE suma 80 posiciones. Sus magnitudes extraordinarias y la degradación de
ambas pérdidas son la combinación que exige investigar.

Se verificó restauración de pesos, Adam, RNG, historial y configuración mediante
regresión determinista en CPU. No apareció un bug demostrable de reanudación.
Eso no prueba que el estado original de época 11 estuviera bien condicionado:
un archivo legible y finito puede contener un modelo cercano a una región inestable.

## Inspección preparada para checkpoints y registros reales

`src/audit_model.py` no ejecuta optimizer.step. Compara copias estáticas best y
last sobre el mismo lote de train/val: CE, z/logits/h0, saturación de h0, normas
de parámetros/gradientes por bloque, espectros recurrentes, sensibilidad del
decoder a z por posición y dependencia de z original/permutado/cero. También
resume los estados de Adam. Conserva pesos, `.grad`, RNG y modos train/eval,
y comprueba SHA-256 del checkpoint antes/después.

```bash
# Ejecutar únicamente sobre copias de archivos estables, cuando termine el run.
python -m src.audit_model --best /copias/best.pt --last /copias/last.pt \
  --processed-dir /content/workshop4_data/processed --split val --batch-size 8 \
  --output /content/audit_best_last.json
python -m src.audit_logs --history /copias/history.csv \
  --gradient-norms /copias/gradient_norms.csv --events /copias/nonfinite_events.jsonl \
  --metadata /copias/metadata.json --output /content/audit_logs.json
```

El probe de modelo usa backend GRU de referencia sin cuDNN, declarado en el
resultado. No reproduce necesariamente los errores/gradientes exactos del
backend CUDA activo y no sirve de benchmark. Sus VJP por posición tampoco son
normas espectrales completas del Jacobiano. VAE se compara en mu, de modo
determinista; esa CE no reemplaza el ELBO Monte Carlo del entrenamiento.
El analizador de logs distingue sesiones/intentos y proporciones de filas
registradas frente a proporciones de todos los batches de una época.

## Rendimiento: diagnóstico y cambios implementados

Los 409 MiB de 15.360 MiB reportados equivalen a aproximadamente 2,7 % de VRAM.
Ni ese porcentaje ni GPU-Util 33–58 % identifican por sí solos un cuello de
botella. La métrica objetivo es ejemplos/s end-to-end con el mismo protocolo.
Los 290 frente a 368–390 segundos se tratan como mediciones suministradas, no
como benchmarks repetidos aquí. El incremento temporal no establece su causa.

Se encontraron costes concretos en el código:

1. Lookup Python por carácter y construcción posterior desde listas: 80 lookups
   por ejemplo más validación por fila y conversión PyArrow.to_pylist.
2. Comprobaciones `.item()` de targets/pérdida, finitud y normas FP64, seguidas
   de lectura de tres métricas escalares: introducen puntos de sincronización.
3. Dos pasadas de norma y conversiones FP64 por seguridad, kernel launches y
   verificación posterior. Se mantienen; no se sustituyen por cálculos inseguros.
4. CSV por lote y, ante eventos enormes, journal+flush/fsync. Es un coste real
   posible, pero no se ha medido su impacto sobre Drive ni se atribuye todo el
   aumento de tiempo a logging.
5. Latente expand con stride temporal cero: PyTorch/backend puede materializar
   copias. No se cambió por GRUCell ni se reimplementaron sus ecuaciones.

Cambios locales, sin aplicar al proceso activo:

- Loader tensorial vectorizado opcional `--tensor-loader`. Valida lo mismo,
  reutiliza vocab/splits y genera los mismos enteros; no cambia el orden.
- Prefetch ordenado acotado `--prefetch-batches 2`, mediante un productor.
  Se propagan errores y se cancela/cierra el lector. Default=0.
- Pinned memory opcional `--pin-memory` y transferencia non_blocking a CUDA.
  No se promete solapamiento de cómputo/copia: se usa el stream habitual.
- Reconstrucción/KL/total se transfieren juntos a CPU, en vez de tres .item()
  independientes. Se verificó equivalencia de métricas y entrenamiento CPU.
- `--gradient-log-every 100` permite muestrear filas ordinarias del CSV;
  defaults mantienen cada lote. Incidencias y gradientes enormes siempre se
  registran, y los agregados por época siguen incluyendo todos los aceptados.
- `--profile-batches N` mide los primeros N (máximo 32) de train y val por
  época: read_prepare, H2D, forward/loss, backward, clipping, Adam y logging.
  CUDA usa eventos y una sincronización de resumen; wall y device no se suman.
  Instrumentación desactivada por defecto y separada de training_config.

Estas opciones de transporte no cambian las matemáticas ni tokens. Se verificó
igualdad bit a bit de pesos, Adam, RNG y métricas en un VAE CPU mínimo. No se
garantiza igualdad bit a bit entre versiones/backend CUDA. Cambiar transporte
con --resume auto se rechaza: requiere una continuación en carpeta nueva y se
registra. No cambiar batch/LR/arquitectura dentro de un entrenamiento existente.

La [guía oficial de pinned memory](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)
advierte que pinnear en la ruta crítica puede resultar más lento. Por eso no
se activó por defecto. Tampoco se activaron AMP, torch.compile, Adam fused,
cuDNN benchmark o algoritmos no deterministas. Cada alternativa necesita un
benchmark y revisión de equivalencia; AMP además requiere precaución por la
inestabilidad ya observada.

## Benchmarks medidos y disponibles

**Medido localmente, solo componente CPU**: Python 3.12.10, PyTorch 2.15.0+cpu,
Windows 11, un hilo. Mismas primeras 512 filas de train, mediana de 3 repeticiones
de 50 codificaciones, incluyendo construcción de x/y tensoriales. Excluye
Parquet, H2D, modelo y Drive. Datos completos: `audit/evidence/data_encoding_cpu.json`.

| Batch | Legacy ms/batch | Tensorial ms/batch | Razón del componente CPU |
| ---: | ---: | ---: | ---: |
| 128 | 3,098 | 0,482 | 6,42x |
| 512 | 11,629 | 0,994 | 11,70x |

**No se midió throughput de Tesla T4 ni una mejora de época completa.** No se
convierten estas razones de CPU en aceleraciones de GPU.

`src/benchmark_ae.py` prepara trials separados para 128/256/512: modelo y Adam
nuevos con igual semilla por trial, warmup 2 y medición 5 pasos (máximo 10).
No carga checkpoints ni produce métricas académicas. Compara profiling off/on,
ejemplos/s end-to-end, intervalos de cómputo, etapas y VRAM pico.

```bash
# SOLO en sesión GPU ociosa, después de finalizar ae_z32; nunca al lado del run.
python -m src.benchmark_ae --model ae --device cuda --synthetic \
  --batch-sizes 128 256 512 --warmup 2 --steps 5 \
  --output /content/bench_ae_compute_t4.json
python -m src.benchmark_ae --model ae --device cuda \
  --processed-dir /content/workshop4_data/processed --tensor-loader \
  --batch-sizes 128 256 512 --warmup 2 --steps 5 \
  --output /content/bench_ae_tensor_t4.json
```

Repetir el segundo sin --tensor-loader para baseline y luego con --pin-memory
si interesa, siempre en salidas distintas y sin cambiar el entrenamiento activo.
El benchmark real lee batches directos de train, sin la ventana shuffle del
trainer, y su logging usa StringIO, **no Drive**. Para conocer el coste real de
Drive se necesita el perfil del entrenador en una ejecución nueva autorizada.
Las mediciones cortas tienen ruido y no demuestran convergencia al cambiar batch.
Los batch sizes son comparaciones independientes, no fases de un mismo run.

## Matriz mínima y protocolo de anomalías

`config/experiments.yaml` conserva doce definiciones, pero añade recommended;
`--recommended` y el notebook seleccionan la matriz de cinco. Entrenar requiere
habilitar explícitamente RUN_TRAINING, que se mantiene en False.

| Experimento de representación | Objetivo |
| --- | --- |
| AE z32 | Referencia principal; conservar best8 existente |
| AE z64 | Comparación de dimensión latente |
| VAE z32 beta 0.5 | Menor regularización relativa |
| VAE z32 beta 1.0 | ELBO estándar |
| VAE z32 beta 4.0 | Mayor regularización relativa |

AE z128 y VAE z64/z128 quedan opcionales, sin ejecución automática. No se
reinicia ae_z32 en esta auditoría. La matriz reduce la grilla propuesta, pero
la adecuación académica final depende de mostrar las comparaciones requeridas.

El filtro normal_class ya se aplica en train/val y se probó. Los cinco runs
generales no sustituyen entrenamientos adicionales normal-only: deben tener
otros nombres y carpetas, clase normal definida usando train y evaluación
mixta posterior. En Shakespeare y es el siguiente carácter; designar su clase
mayoritaria como normal es una definición operacional que debe justificarse,
no una etiqueta natural de anomalías de texto. No se ha implementado aún el
ajuste/evaluación del umbral en Fase 4: deberá usar val, nunca test. No hay AUC
de anomalías calculado aquí ni comparación directa válida con AUC multiclase.

## Actualizar Colab y decisiones antes de nuevos modelos

1. Dejar terminar el proceso activo. No reinstalar paquetes, volver a ejecutar
   bootstrap, montar otro proyecto sobre el actual ni reemplazar archivos de
   Drive o /content que use esa sesión.
2. Al finalizar, preservar directorio completo, best/last, CSV, journal,
   metadata y versión ZIP usada. Copiar archivos estáticos para audit_model y
   audit_logs; no editar ni sustituir los originales.
3. Subir el ZIP auditado a un nombre nuevo y extraer en otra carpeta /content.
   Mantener dataset/vocab/splits. Usar outputs nuevos para perfiles/benchmarks
   e inspector. No arrancar --all ni reiniciar ae_z32 por actualizar el ZIP.
4. Comparar best8/last real: bloque que domina gradientes, sensibilidad por
   horizonte, z y saturación, dependencia de entrada, Adam finito y magnitud
   de actualizaciones. Sin eso queda abierta la causa primaria.
5. Elegir transporte y batch de los próximos VAE según el benchmark T4 y luego
   comprobar calidad en una ejecución nueva autorizada. Cambiar batch altera
   número de actualizaciones y dinámica; no basta con que aumente ejemplos/s.
6. Evaluar si hace falta un experimento nuevo de estabilidad: inicialización
   ortogonal por gate o limitar/normalizar la entrada latente son candidatos,
   no correcciones demostradas. Añaden cambios de arquitectura/dinámica y no
   deben aplicarse al checkpoint existente como si fueran la misma ejecución.
   No se implementaron automáticamente aquí. Reducir lr, reiniciar Adam o
   truncar backward también requiere un protocolo nuevo y autorización.

Para VAE seguir observando reconstrucción y KL por separado y colapso posterior.
No activar annealing, clamp de logvar o pérdidas con otra reducción en medio
de un run. La estabilidad del AE debe investigarse antes de lanzar la grilla.
La [nota de reproducibilidad PyTorch](https://docs.pytorch.org/docs/2.14/notes/randomness.html)
explica las limitaciones de equivalencia entre hardware/versiones; no se promete
que CPU local 2.15 y Colab 2.14.1+cu130 reproduzcan números idénticos.

## Validación y limitaciones finales

Las pruebas incluyen contratos de datos y Unicode/UNK, equivalencia de tensores,
salidas y gradientes, clipping FP64 extremo, protección de NaN/Inf, Adam/RNG y
checkpoint legado, prefetch ordenado/cancelable, profiling acotado, matriz de
cinco y herramientas de auditoría sin efectos sobre las fuentes. El resultado
consolidado fue **77 pruebas aprobadas, cero errores y cero omisiones**, en CPU.
Se verificaron sintaxis/celdas no ejecutadas y los ocho artefactos históricos
locales conservaron sus hashes. Registro: `audit/evidence/validation.json`.

El paquete de esta auditoría es `exports/workshop4_colab_audited.zip`, generado
después de validar. El ZIP excluye Parquet, checkpoints y runs; incluye código,
informe y evidencia. Los ZIP anteriores se conservan para identificar versiones.

Persisten tres límites materiales: no hay checkpoints/logs de producción
locales, no se probó una GPU en esta sesión y no se demostró una modificación
que restaure la reconstrucción del modelo deteriorado. Las herramientas y la
evidencia permiten decidir el siguiente experimento sin inventar una causa.
