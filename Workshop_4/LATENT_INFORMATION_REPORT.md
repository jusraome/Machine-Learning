# Fase 3 — Diagnóstico de dependencia informativa del latente

Fecha: 14 de noviembre de 2026. Evidencia local CPU y resultados T4 comunicados
por el usuario, identificados por separado. No se ejecutó entrenamiento completo.

## Resultado y alcance de la conclusión

**En un experimento nuevo de 24 actualizaciones, tanto identity como latent_tanh
aprovechan muy poco la identidad de la secuencia codificada en z sobre el prefijo
fijo de 128 secuencias de validación.** Un baseline de frecuencias ajustado con
todo train reconstruye mejor. Intercambiar códigos reales apenas cambia pérdida
y probabilidades; ambas variantes predicen espacio en las 10.240 posiciones.

Esto respalda un predictor casi independiente del ejemplo y compatible con un
prior marginal todavía mal calibrado. No demuestra que haya aprendido exactamente
las frecuencias: la TV respecto al baseline sigue alrededor de 0.10. Tampoco
demuestra pérdida de información *previamente aprendida*: el modelo aleatorio
inicial ya obtenía una ventaja reconstructiva casi nula del código correcto.
La evidencia distingue **caída de respuesta a z** de **aprendizaje y posterior
pérdida de información útil**, que estos pilotos no establecen.

No se propone iniciar AE z64 ni VAE todavía. Las alternativas de arquitectura al
final de este informe son hipótesis posteriores a estos diagnósticos, sin
implementación ni selección automática.

## Evidencia comunicada y evidencia medida

El usuario comunicó cinco pilotos T4 con 12.288 ejemplos por variante:

| Variante | Val nats/secuencia | Saturación decoder | Sensibilidad antigua |
| --- | ---: | ---: | ---: |
| reference_b128 | 253.626 | 0.5971 | 0.002738 |
| batch256 | 253.909 | 0.5806 | 0.002152 |
| batch512 | 256.523 | 0.3787 | 0.002874 |
| lr0003 | 253.671 | 0.3945 | 0.001507 |
| latent_tanh | 253.479 | 0.4139 | 0.000608 |

También comunicó clipping en el 100 % de actualizaciones, ninguna parada de
seguridad y una trayectoria de sensibilidad de referencia 0.29017 → 0.06071
(12 updates) → 0.002836 (24). Estos números se preservan como información del
usuario en `audit/latent_information_phase3/user_reported_context.json`.

El runner anterior no guardaba pesos de pilotos. Los checkpoints de producción
best8/last30 y de los cinco pilotos T4 no están disponibles localmente para esta
evaluación. Los artefactos smoke locales no se presentan como esos modelos.
La nueva comparación CPU usa **3.072 ejemplos y 24 actualizaciones por modelo**,
no 12.288. Su prefijo/permutación difiere de la reserva anterior de 12.288.
No es una reproducción numérica de aquellos cinco pilotos.

## Baselines sin fuga de validación

Se recorrieron las **2.997.217 ventanas de train**, contabilizando sus
**239.777.360 posiciones x**. No se ajusta con val, test, y ni texto raw.
Las ventanas solapadas cuentan tantas veces como en el objetivo del AE; no se
pretende estimar la distribución de caracteres únicos del texto original.

Para V=76 y N posiciones, el baseline global usa
`p(c) = (count_train_x(c) + 0.5) / (N + 0.5*V)`.
Se añade un baseline por posición
`p_t(c) = (count_train_x(t,c) + 0.5) / (n_train + 0.5*V)`.
Así se puede detectar aprendizaje de un prior posicional sin información del
ejemplo. El suavizado 0.5 está fijado antes de observar val. Incluye UNK=0 y no
introduce PAD ni máscara; train contiene cero UNK. La API admite alpha=0 y
señala explícitamente pérdida indefinida si val contiene categorías sin soporte,
en vez de serializar Infinity como un número válido.

Todos los resultados siguientes usan exactamente las mismas primeras **128
ventanas val, 80 posiciones por ventana**. El objetivo es CE de reconstrucción
contra x: suma temporal y media por ejemplos, en nats/secuencia. Dividir entre
80 produce nats/carácter. No se compara con CE contra el siguiente carácter y.
La CE de las comparaciones nuevas y sus agregaciones se calculan en FP64 sobre
logits FP32. `legacy_first_batch` conserva la CE histórica en FP32.

| Predictor | Nats/secuencia | Nats/carácter |
| --- | ---: | ---: |
| Marginal global de todo train | 252.388629 | 3.154858 |
| Marginal por posición de todo train | 252.385852 | 3.154823 |
| AE identity, paso 24 | 258.726156 | 3.234077 |
| AE tanh, paso 24 | 258.325130 | 3.229064 |

El baseline posicional apenas cambia la pérdida en esta muestra. Identity pierde
6.337527 nats/secuencia contra la marginal global; tanh pierde 5.936501. La
cercanía de los valores T4 comunicados a este baseline es sugerente, pero sin
verificar sus fingerprints/pesos no constituye una comparación pareada formal.

## Protocolo corto ejecutado

Dos modelos AE z32 nuevos: embedding 32, hidden GRU 128, una capa, dropout 0,
Adam lr=0.001, batch 128, semilla 42. Comparten pesos iniciales bit a bit, Adam
nuevo y el mismo prefijo train de 3.072 ejemplos, permutado una vez con un
generador local. El hash inicial compartido es
`3f25b092351ce8796dbab6f04a3d9a8c2f8a235c105527bc86ac34865f441b07`.

Solo cambia `latent_activation`: identity frente a tanh antes de **ambas** rutas
del decoder (estado inicial y entrada repetida). No se cambia el modelo original,
la pérdida, ni el entrenamiento de producción. Se midió en pasos 0, 1, 3, 6, 12
y 24, siempre sobre el mismo val. Entorno: PyTorch 2.15.0+cpu, sin CUDA; pesos y
operaciones del modelo FP32, métricas/normas/escalado de clipping FP64.

Los dos modelos completaron 24 actualizaciones y necesitaron clipping en 24/24;
norma máxima previa 90.08384 identity y 85.21762 tanh. Se conservó clipping global
a 5.0, verificación posterior, checks de pérdida/z/gradientes/pesos/Adam finitos,
límite raw-z=100, norma raw=1e8 y paradas por deterioro. No hubo paradas.
Los snapshots son JSON de diagnóstico, **no checkpoints** ni continuaciones.

La intervención principal intercambia z con desplazamiento cíclico de media
población: una biyección sin puntos fijos, incluso con B impar. Se aplica a la
población completa antes de comparar, sin cambiar objetivos ni posiciones.
En estos datos hay 0 % de secuencias idénticas donante/receptor y 5.996 % de
acuerdo por posición. Esto reduce el problema de intercambiar ventanas vecinas,
aunque no garantiza diversidad semántica entre hablantes.

## Intervenciones: identidad del código frente a condición común

| Medida, paso 24 | Identity | Tanh |
| --- | ---: | ---: |
| CE original, nats/secuencia | 258.726156 | 258.325130 |
| CE swap − CE original, nats/secuencia | 0.007146 | 0.001925 |
| CE swap − CE original, nats/carácter | 0.00008933 | 0.00002406 |
| CE cero − CE original, nats/secuencia | 8.958764 | 7.842107 |
| TV media original/swap | 0.001919 | 0.000655 |
| RMS cambio de logits centrados | 0.005661 | 0.001482 |
| Fracción argmax que cambia al intercambiar z | 0 | 0 |
| TV original/marginal global | 0.102110 | 0.103905 |
| Saturación decoder, primer batch val | 0.421219 | 0.114792 |

El daño de swap es diminuto comparado con el exceso de pérdida respecto al
baseline. Sus SD pareadas son 0.05485 y 0.01544 nats/secuencia: se guardan todos
los deltas individuales y su dispersión, sin tratar las ventanas como muestras
IID ni anunciar significación estadística. El argmax es espacio en todas las
posiciones para todos los ejemplos. Esta observación por sí sola sería
insuficiente; TV/JS, pérdidas y diversidad de probabilidades corroboran la escasa
dependencia útil del ejemplo en el paso 24.

El efecto de z=0 es mucho mayor que el de intercambiar códigos reales. Por tanto,
no se afirma que el decoder ignore absolutamente z: puede usar una componente
común como condición/bias. Lo que apenas aprovecha aquí es **qué ejemplo produjo
el código**. Cero puede quedar fuera de la distribución aprendida; esa pérdida
no mide directamente información específica de la secuencia.

La dispersión entre ejemplos de z (RMS centrado) al final es 0.062658 identity
y 0.012324 tanh, mientras la variación de probabilidades entre ejemplos es muy
pequeña (varianza media 8.04e-9 y 1.13e-9). Esto no prueba una causa única en el
encoder o decoder. Tanh reduce saturación, pero no recupera utilidad informativa.

## Por qué la sensibilidad histórica no basta

Se conserva la fórmula histórica:
`||decode(reverse(z)) - decode(z)||₂ / max(||decode(z)||₂, 1e-12)`.
El runner la calcula sobre el primer batch fijo de 32, igual que el piloto, y
el módulo también sobre toda la población val. Revertir un batch impar deja un
punto fijo; la nueva intervención principal evita ese problema.

Esta razón depende de offsets de logits que softmax ignora, de su escala, de la
dispersión de z y del emparejamiento. No mide daño reconstructivo ni información
mutua. Un cambio grande también puede ser una respuesta arbitraria sin utilidad.

Un ejemplo **medido con la misma población de 128 y la misma permutación nueva**:

| Identity | Paso 0 | Paso 12 | Paso 24 |
| --- | ---: | ---: | ---: |
| Cambio L2 relativo de logits por swap | 0.306291 | 0.094219 | ~0.003 |
| RMS de logits originales | 0.059216 | 0.883493 | 2.011072 |
| RMS de cambio de logits | 0.018137 | 0.083242 | ~0.006 |
| RMS cambio centrado | 0.018031 | 0.077178 | 0.005661 |
| TV de probabilidades por swap | 0.007027 | 0.036263 | 0.001919 |

Entre 0 y 12 baja la razón relativa mientras **aumentan** el cambio absoluto y
TV. La caída temprana de la razón no equivale a pérdida de respuesta; el
denominador crece. Tanh muestra el mismo desacople. Al paso 24 sí se reducen
también los cambios absolutos/probabilísticos y el daño útil de swap es mínimo.

Las pruebas sintéticas incluyen dos contraejemplos independientes: añadir una
constante común a logits reduce la razón relativa sin modificar softmax ni el
cambio centrado; y una razón <0.002 coexiste con TV>0.45 y predicciones invertidas.
También prueban un decoder perfectamente informativo y otro invariante a z,
para comprobar que el nuevo diagnóstico distingue ambos.

Se registran RMS/L2 raw y centrados, TV, JS en nats, entropía, exactitud,
histogramas, diversidad entre ejemplos, dispersión/rango efectivo del latente,
CE pareada y comparación directa de distribuciones con ambos baselines.
No se presenta ninguna de estas magnitudes como estimador de información mutua.

## Diagnósticos por posición

Se guardan las 80 posiciones, con índices JSON 0–79 y gráficos 1–80. Los tercios
son disjuntos y exhaustivos: inicial 1–27, intermedio 28–54, final 55–80.

| Swap − original al paso 24, nats/carácter | Identity | Tanh |
| --- | ---: | ---: |
| Inicial | 0.00005444 | 0.00002563 |
| Intermedio | 0.00005529 | 0.00001406 |
| Final | 0.00016090 | 0.00003282 |

No aparece un tercio con ventaja reconstructiva grande del código correcto. El
promedio por posición individual oscila aproximadamente entre −0.001187 y
0.001055 identity, y −0.000241 y 0.000369 tanh. No se concluye que exista un
segmento informativo solo por signos positivos locales: se muestran los puntos
individuales y las distribuciones pareadas. `positions.csv` y el notebook permiten
revisar primeros, intermedios y últimos caracteres sin esconderlos en una media.

## Alternativas posteriores al diagnóstico

Las observaciones anteriores justifican formular dos pruebas futuras, sin
declarar resuelta la arquitectura ni entrenarlas en esta fase:

1. **Posición explícita y ruta residual z→logits.** Añadir embeddings de posición
   y una cabeza acotada que combine z y posición y contribuya directamente a los
   logits, manteniendo la GRU. Acorta la ruta hacia la pérdida y permite comprobar
   si la dinámica recurrente atrae las salidas a un prior compartido. Una simple
   proyección constante Wz puede aprender solo frecuencias del ejemplo; por eso
   la interacción con posición debe ser explícita. Una conexión adicional no
   obliga al decoder a usar información: exigir mejora frente a ambos baselines
   y daño pareado de swap reproducible, además de TV.
2. **Decoder de consultas posicionales desde el bottleneck.** Predecir posiciones
   mediante una red compartida con consultas/embeddings posicionales y z como
   única fuente de contenido, reduciendo el camino recurrente de 80 pasos.
   Puede aprender solo un prior posicional y perder capacidad de modelar orden;
   debe compararse con presupuesto y parámetros controlados. No añadir acceso
   directo a x ni teacher forcing que permita esquivar el cuello de botella.

Antes de escoger una: evaluar checkpoints históricos compatibles con el mismo
protocolo; repetir el diagnóstico con otras semillas y subconjuntos de val
predefinidos que incluyan más hablantes, manteniendo intercambios informativos.
La evaluación de checkpoints está implementada y no restaura Adam. No se
reconstruye retrospectivamente un piloto sin sus pesos.

## Reproducción y entregables

Desde `Workshop_4`, revisar un plan sin efectos de entrenamiento:

```bash
python -m src.latent_diagnostic_runner \
  --processed-dir /ruta/processed --device cuda \
  --settings config/latent_information_phase3.json \
  --output-dir /ruta/nueva/diagnostico
```

Añadir `--execute` para ajustar/evaluar los baselines. Con
`--checkpoint best=/ruta/best.pt --checkpoint last=/ruta/last.pt` se evalúan AE
compatibles en lectura; con `--short-probe` se añaden exclusivamente las dos
pruebas nuevas de hasta 24 updates. Cada invocación exige una salida inexistente.
El CLI supervisa un proceso hijo con timeout global +20 s; Colab añade otro de
650 s. Una operación en curso puede exceder el límite blando hasta ese corte.
Un corte duro deja artefactos parciales, que no cuentan como completados.

Notebook nuevo: `notebooks/Workshop_4_Latent_Information_Colab.ipynb`.
ZIP nuevo: `exports/workshop4_colab_latent_phase3.zip`. Conserva los ZIP previos.
El notebook viene sin resultados y con EXECUTE=False/RUN_SHORT_PROBE=False.
Solo usa el nuevo runner; no invoca el trainer completo ni la matriz de 12
experimentos. Se incluyen código/config/tests/docs y metadatos W1 de lectura,
sin Parquet, pesos ni entornos. No instala dependencias automáticamente.

Resultados locales nuevos: `audit/latent_information_phase3/`:

- `local_evaluation/`: baseline completo y evaluación sobre val, cero updates.
- `local_short_identity_tanh/`: datos/hash/plan, métricas por snapshot y resultados.
- `derived/`: CSV comparativos y figuras de trayectoria/posición.
- `tests_final.json`, `validation.json`, `package_validation.json`: pruebas,
  hashes y verificación del paquete. Los 18 archivos protegidos conservaron sus
  hashes. Las figuras se revisaron visualmente.

Se aprobaron **114 pruebas unitarias/integración**, cero errores y cero omisiones,
incluidas 24 nuevas. Verifican baselines train-only, unidades, posiciones,
permutaciones, invariancia de modos/RNG/pesos/.grad, checkpoints inmutables,
límites, clipping FP64 y que no se llame Adam.step con gradientes inválidos.
Los tests existentes incluyen pequeños smoke sintéticos; no son ejecuciones
de la matriz de experimentos. No se hizo commit ni push.

## Limitaciones que siguen abiertas

- Val es un prefijo fijo de 128 ventanas, con posible correlación por solapamiento
  y concentración de hablantes; no equivale a validación completa representativa.
- Train de los pilotos es un prefijo fijo. Los nuevos usan 3.072 ejemplos por
  modelo, una semilla y CPU; los comunicados usaron 12.288 y T4. No extrapolar
  diferencias pequeñas ni velocidades entre estas ejecuciones.
- Una permutación no explora todas las intervenciones posibles. Se registran
  sus índices y semejanza de donantes; la API admite otra biyección sin puntos
  fijos. TV pequeña solo en esta población no prueba independencia universal.
- Las medidas no estiman información mutua ni prueban un colapso global formal,
  ni identifican causalmente qué componente inicia el fenómeno.
- El modelo inicial no demuestra reconstrucción informativa aprendida; hablar
  de pérdida posterior de información previamente útil excedería la evidencia.
- Saturación es la fracción de estados con |h|≥0.99, no probabilidades de gates.
  Menor saturación y ausencia de NaN no garantizan información útil.
- **Ninguno de estos pilotos demuestra estabilidad durante 30 épocas.** Las
  protecciones numéricas FP64 se conservan, pero clipping en todos los updates
  y terminar el presupuesto no validan calidad ni estabilidad de largo plazo.
