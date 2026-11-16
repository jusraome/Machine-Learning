# Plan de pruebas cortas de estabilidad — AE GRU

## Alcance y estado de la evidencia

Este paquete prepara experimentos nuevos y acotados. No modifica best.pt de
época 8, last.pt de época 30, sus optimizadores, CSV, registros ni datos W1.
No reanuda ni reinicia ae_z32. Un checkpoint histórico se utiliza únicamente
como modelo de evaluación, nunca para inicializar el optimizador de un piloto.

Los siguientes resultados proceden de la solicitud del usuario, no de una
lectura nueva de Drive: entrenamiento terminado en 30 épocas; best8 val=168.6650,
last30 val=249.5645; mayor magnitud de z, mayor saturación y menor sensibilidad
del decoder en last30; clipping en todos los lotes desde época 12, sin rechazos.
La causa inicial sigue sin estar demostrada. No se anuncia una reparación de
la estabilidad durante 30 épocas.

También se comunicaron benchmarks T4 de cinco lotes:

| Batch | Loader original, ejemplos/s | Loader tensorial, ejemplos/s |
| ---: | ---: | ---: |
| 128 | 6 658 | 7 189 |
| 256 | 11 267 | 13 162 |
| 512 | 15 023 | 17 615 |

Justifican conservar y probar el loader tensorial, no escoger batch 512 sin
comparar calidad/dinámica. No garantizan el tiempo de una época ni estabilidad.

## Revisión de implementación e hipótesis

El decoder ya procesa los 80 pasos con una única nn.GRU. El vector z participa
en dos rutas: la proyección tanh del estado inicial y la entrada repetida de cada
paso. El tanh del estado inicial no limita la segunda ruta. La proyección lineal
del encoder puede aumentar su magnitud, y los Jacobianos recurrentes pueden
amplificar el backward aun con estados ocultos acotados.

No se encontró un error demostrado de broadcasting, objetivo, doble backward
o softmax duplicado: se mantiene CE sobre logits `(B,80,76)`, objetivos x y
suma temporal seguida de media por ejemplos. UNK=0 no se ignora; no existe PAD.
La revisión anterior y sus tests verificaron la restauración de Adam/RNG y el
clipping global FP64. Adam mantiene momentos históricos por coordenada: una
norma clippeada a 5 no implica una cota idéntica para cada actualización de pesos.

Las correlaciones de last30 apoyan probar el control de z, pero no demuestran
si el crecimiento de z inició el problema o fue una consecuencia. Tampoco se
demostró una inicialización incorrecta: se conserva la inicialización original.
No se cambia clipping global por clipping por bloque ni se reinicia Adam de un
checkpoint para intentar explicar la degradación.

## Matriz de un factor por variante

Configuración: `config/stability_experiments.yaml`. Todos los pilotos utilizan
semilla 42, AE z32, embedding 32, hidden 128, una capa GRU, dropout 0, Adam,
CE original y clipping global FP64 a 5.0. El loader tensorial se mantiene
constante en todos ellos y puede desactivarse en el plan completo, no solo
para favorecer una variante. Todas las variantes parten de pesos nuevos e
idénticos, y de un Adam nuevo sin estado; se registran sus hashes iniciales.

| Variante | Batch | LR | Latente | Factor respecto al control |
| --- | ---: | ---: | --- | --- |
| reference_b128 | 128 | 0.001 | identity | Ninguno |
| batch256 | 256 | 0.001 | identity | Batch |
| batch512 | 512 | 0.001 | identity | Batch |
| lr0003 | 128 | 0.0003 | identity | Learning rate |
| latent_tanh, opcional | 128 | 0.001 | tanh | Arquitectura |

`tanh` se aplica una sola vez a la proyección z antes de ambas rutas del
decoder. No añade parámetros, conserva la inicialización y declara
`latent_activation=tanh` en hparams, pero **sí cambia la arquitectura y la
función aprendida**. Requiere seleccionar su nombre y --include-architecture;
--all no la ejecuta. Identity preserva hparams/state_dict y resultados legados.
VAE permanece sin cambios. El trainer de 30 épocas no adopta esta variante
automáticamente; si resulta útil habrá que definir un experimento completo nuevo.

Se registran z consumido y z raw. Tanh puede esconder el crecimiento de raw z
tras una salida acotada o producir derivadas pequeñas; por eso el límite de
magnitud también inspecciona raw z. Un z acotado no garantiza Jacobianos GRU
bien condicionados ni calidad de reconstrucción.

## Dos presupuestos separados para comparar con justicia

Cada combinación variante/presupuesto es un trial independiente, con pesos y
Adam reiniciados desde la misma semilla. No se aumenta el batch a mitad de un
trial. `summary.json` incluye comparaciones descriptivas solo si los trials
terminaron y comparten inicialización y presupuesto cumplido.

| Presupuesto | Batch 128 | Batch 256 | Batch 512 |
| --- | --- | --- | --- |
| Actualizaciones iguales | 24 updates, 3 072 ejemplos | 24 updates, 6 144 ejemplos | 24 updates, 12 288 ejemplos |
| Ejemplos iguales | 96 updates, 12 288 ejemplos | 48 updates, 12 288 ejemplos | 24 updates, 12 288 ejemplos |

Las variantes LR/tanh usan los conteos de batch 128. Un batch mayor cambia la
cantidad de actualizaciones y el ruido del gradiente; no se atribuye una mejora
de pérdida solamente a su mayor throughput. Los datos se precargan desde un
prefijo acotado de train, con una permutación común según la semilla. La reserva
común se determina por el plan completo, incluso al ejecutar variantes/modos
en comandos separados; su hash y el de val quedan registrados.

Se utilizan las particiones existentes, nunca test. El prefijo de train de
12 288 ejemplos y los primeros 128 de val son un conjunto diagnóstico reducido,
**no una muestra representativa garantizada** de todos los hablantes. Esto no
reemplaza la validación completa posterior. Debe mantenerse el mismo plan/datos
para comparar: cambiar el tamaño de la reserva altera su permutación.

## Métricas e interpretación

`metrics.csv` registra filas train/validation/final con updates y ejemplos:

- CE train acumulada ponderada por ejemplos y CE del mismo subconjunto val.
- Normas pre-clipping media/máximo, máximo posterior y fracción clippeada.
- Magnitud máxima y RMS de z, magnitud raw antes del control.
- Fracción `abs(hidden)>=0.99` en la secuencia del decoder y en h0: proxies
  de saturación del estado, no probabilidades observadas de los gates GRU.
- Cambio L2 relativo de logits al invertir el orden de z entre ejemplos,
  calculado sobre el primer batch fijo de validación. Con un solo ejemplo
  la permutación no es informativa; usar validation_batch_size>=2.
- Ejemplos/s del tramo de entrenamiento y ejemplos/s end-to-end del trial,
  tiempo total y VRAM máxima asignada (null en CPU).

Diagnósticos de saturación/sensibilidad se toman en validación inicial, cada
12 updates y al final; no se presentan como si se midieran en cada batch de
train. Los campos no medidos antes del primer update quedan null/vacíos.
Las pérdidas de pilotos recién inicializados no deben compararse como si
hubieran recibido los millones de ejemplos de best8. El checkpoint histórico
opcional se evalúa sobre el mismo val reducido y se guarda por separado; su
pérdida sobre ese prefijo tampoco equivale al 168.6650 de toda su validación.

La instrumentación y la evaluación consumen tiempo. El tramo de entrenamiento
excluye diagnósticos de val y escritura de métricas; end-to-end los incluye.
La precarga no se incluye en throughput por trial, pero sí en tiempo total de
suite. No extrapolar esas velocidades ni los benchmarks de cinco lotes a épocas.

## Límites y paradas seguras

Defaults: 120 segundos por trial, 600 por suite, 600 updates y 100 000 ejemplos
de entrenamiento acumulados como máximos globales. Las evaluaciones también
se acotan: máximo 8 192 ejemplos val previstos por trial y 65 536 por suite.
El modo por ejemplos tiene como máximo 96 updates; el modo por updates, 24.

Hay techos absolutos que el YAML/CLI no pueden ampliar: 128 updates y 65 536
ejemplos por trial, 180 segundos por trial, 900 por suite, 1 024 updates y
262 144 ejemplos globales, val de como máximo 512 ejemplos por evaluación.
Los chequeos de plazo ocurren entre etapas; una operación GPU/IO en curso no
se puede interrumpir de forma segura dentro del proceso. Por eso el CLI
supervisa un hijo y lo termina al superar max_seconds_total+20 segundos de
arranque. El notebook añade un timeout exterior de +30 segundos. Tras un corte
duro solo hay resultados parciales; no son trials completados.

Las siguientes paradas no modifican pesos históricos ni elevan límites de
lotes omitidos:

- NaN/Inf de pérdida, gradiente, modelo, Adam o diagnósticos: parar; nunca
  ejecutar Adam con gradientes inválidos. El piloto no omite esos lotes.
- Norma raw superior a 1e8 o magnitud raw/consumida de z superior a 100:
  detener ese trial antes de actualizar. Son guardas diagnósticas conservadoras,
  no umbrales demostrados óptimos ni una prueba de causa inicial.
- Validación mayor que 3 veces su valor inicial en dos evaluaciones seguidas,
  o train mayor que 6 veces ese valor inicial: parar por deterioro extremo.
- Tiempo, datos insuficientes, presupuesto incomparable o presupuesto global:
  marcar stopped/incompleto y excluir de una comparación como trial completado.

Estos criterios permiten detectar fallos tempranos, no todos los fallos tardíos.
El incidente original apareció tras millones de ejemplos; un piloto de miles
de ejemplos sin errores **no demuestra estabilidad durante 30 épocas**.

## Ejecución explícita en Colab

Nuevo notebook: `notebooks/Workshop_4_Stability_Colab.ipynb`. Paquete:
`exports/workshop4_colab_stability_plan.zip`. Conservar también todos los ZIP y
carpetas históricos; extraer el nuevo a otra carpeta y usar una salida nueva
de Drive. El runner rechaza cualquier directorio de salida ya existente.

1. Seleccionar T4, montar Drive y cargar el ZIP. Ajustar las rutas de Parquet
   existentes; el controlador solo prepara train/val, sin descargar Shakespeare.
2. Si se desea referencia histórica, indicar best.pt de época 8. El programa
   valida vocabulario/contrato, carga solo el modelo y comprueba su SHA-256.
   Nunca restaura Adam, ni toma last30 como inicialización del piloto.
3. Con EXECUTE=False revisar variantes, presupuesto y límites. No se entrena
   ni se crean carpetas de resultados en este modo.
4. Primera acción recomendada: seleccionar solo reference_b128 y ambos
   presupuestos para comprobar el circuito. Después, con una **salida nueva**,
   ejecutar los cuatro controles no arquitectónicos. Tanh se prueba solo si
   se selecciona explícitamente y se activa INCLUDE_ARCHITECTURE.
5. Habilitar EXECUTE para iniciar. Revisar status/reason y comparaciones en
   summary.json y metrics.csv; no escoger un modelo porque una celda terminó
   sin excepción. No hay ganador ni entrenamiento largo automáticos.

Equivalente CLI desde el Workshop_4 extraído, primero sin --execute:

```bash
python -m src.stability_experiments \
  --config config/stability_experiments.yaml --all --budget both \
  --processed-dir /content/workshop4_data/processed --device cuda \
  --output-dir /content/drive/MyDrive/Workshop_4/stability/pilot_nuevo
# Tras revisar el plan, repetir añadiendo --execute.
```

Para la variante arquitectónica, elegir --variant latent_tanh y
--include-architecture; incluir también --variant reference_b128 en la misma
suite si se quiere comparación automática contra el control. --reference-checkpoint
solo añade una evaluación histórica, no una continuación de entrenamiento.
No se escriben checkpoints de pilotos: no existe --resume y cada trial exige
una inicialización nueva; el usuario conserva configuraciones, métricas y hashes.

## Decisiones tras los resultados

Primero comparar calidad y diagnósticos con el control para cada presupuesto.
Si un trial se corta, falta control o no cumplió el presupuesto, no interpretar
su menor pérdida/tiempo como victoria. A continuación repetir los candidatos
con otras semillas y muestras diagnósticas explícitas, en salidas nuevas.
Solo después proponer una comprobación más larga y validación completa, con
autorización y sin sustituir best8. No decidir únicamente por VRAM o ejemplos/s.

Esta implementación estudia AE. La configuración elegida no demuestra
estabilidad de VAE: habrá que repetir un protocolo específico para cada beta,
observando KL y reconstrucción por separado. No se aplica tanh a mu/logvar ni
se altera la distribución variacional automáticamente. El entrenamiento
normal-only de anomalías sigue separado del análisis general.

## Validación local

Se verifican referencia bit a bit, control diferenciable, hparams/checkpoints
legados, controles de un factor, igualdad de inicialización/datos, presupuestos
comparables, fuentes inmutables, best solo para evaluación, NaN sin Adam.step,
plazos y deterioro. Además se ejecutó un smoke del CLI supervisado sobre
Shakespeare real: **dos updates CPU, cuatro ejemplos de train**, con modelo
reducido y cuatro ejemplos val, en audit/stability_validation/real_cpu_smoke.
Ese smoke comprueba integración; sus pérdidas no seleccionan configuraciones.

Resultado consolidado: **90 pruebas aprobadas, cero errores y cero omisiones**
en CPU. Los 14 archivos protegidos de datos/checkpoints/registros conservaron
sus hashes. Se validaron sintaxis y celdas sin ejecutar de ambos controladores.
Evidencia: `audit/stability_validation/validation.json`. Los resultados reportados
por el usuario se identifican en `user_reported_context.json`, en esa carpeta.
Todavía no se ejecutaron los pilotos comparativos de esta versión en T4.
