# Auditoría fase 4 — deterioro del autoencoder original

## Plan y controles

Esta entrega establece un procedimiento de diagnóstico; **no inicia un
entrenamiento completo**. Se conserva la arquitectura AE original z32,
la CE de reconstrucción, el clipping global FP64 a 5.0 y los históricos.
No se ejecutan AE z64, VAE ni la matriz de doce experimentos.

Orden de trabajo:

1. Verificar los cuatro logs y ambos checkpoints en Drive tras montarlo.
2. Copiar los logs a una carpeta nueva, verificar hashes y analizar épocas 8–15.
3. Comparar los endpoints históricos sobre los mismos 128 ejemplos y examinar
   estados/gradientes en lectura cuando los pesos estén accesibles.
4. Revisar el plan de una ventana corta instrumentada, con checkpoints completos
   periódicos y controles pareados de reconstrucción/dependencia de z.
5. Detener la rama ante deterioro según criterios fijados de antemano; conservar
   el mejor estado elegible y la evidencia del primer intervalo problemático.
6. Solo después plantear un contraste controlado de Adam desde un checkpoint
   nuevo y completo; no atribuir causalidad a correlaciones o a un reinicio solo.

La ventana preparada tiene 64 actualizaciones por defecto, evalúa y guarda cada
8, usa batch128, seed42 y Adam lr0.001. Límite duro: 256 actualizaciones
**acumuladas por línea diagnóstica**, 600 s por invocación. Es un circuito para
localizar transiciones cercanas y probar controles, no una reproducción de
cuatro épocas de millones de ejemplos a partir de best8. No hay bucle que repita
ventanas automáticamente ni un modo de entrenamiento largo en este runner.

## Evidencia nueva y actualización de la conclusión

Los siguientes resultados fueron confirmados por el usuario sobre **las mismas
128 secuencias de validación, con 80 posiciones cada una**. Se incorporan como
evidencia confirmada comunicada, sin atribuirles una reevaluación local nueva.

| Métrica | Época 8 | Época 30 |
| --- | ---: | ---: |
| Reconstrucción, nats/secuencia | 160.400189 | 250.129855 |
| Reconstrucción, nats/carácter, derivada | 2.005002 | 3.126623 |
| Baseline marginal train, nats/secuencia | 252.388629 | 252.388629 |
| Ventaja frente a marginal, nats/secuencia | 91.988440 | 2.258774 |
| CE intercambiada − original, nats/secuencia | 359.958162 | 3.640303 |
| TV original/intercambiado | 0.662012 | 0.013355 |

La reconstrucción empeora **55.94 %**. Queda aproximadamente **1.01 %** del daño
por intercambio y **2.02 %** de la TV originales. La época 8 sí demuestra
reconstrucción específica del ejemplo y uso informativo fuerte del latente.
La época 30 conserva utilidad residual: supera la marginal por 2.258774
nats/secuencia y el código alineado aún aporta 3.640303. Se concluye **deterioro
marcado**, no independencia absoluta de z ni colapso formal de información mutua.

Esto amplía la evidencia de fases previas. Los pilotos nuevos aleatorios de
fase 3 no demostraban una etapa inicial de reconstrucción informativa aprendida;
los endpoints históricos ahora sí muestran esa capacidad en época 8 y su
deterioro posterior. Los informes anteriores se preservan como registros de la
evidencia disponible entonces. La **causa raíz sigue sin establecerse**: dos
checkpoints no localizan qué actualización inició el problema.

La comparación diagnóstica no es la validación completa. Por ejemplo, el valor
histórico comunicado para la validación de época 8 era aproximadamente 168.67,
mientras el prefijo de 128 da 160.400189. Son distintos conjuntos/ponderaciones,
no una contradicción ni una mejora nueva. No se mezclan sus series para definir
ratios, umbrales, selección de modelo o significación estadística.

## Fuentes de producción y análisis disponible

Ruta de Drive confirmada por el usuario:

```text
/content/drive/MyDrive/Workshop_4/runs/workshop4_clipping_v2_prueba12/ae_z32/
  history.csv
  gradient_norms.csv
  nonfinite_events.jsonl
  metadata.json
  checkpoints/best.pt   # época 8
  checkpoints/last.pt   # época 30
```

No se presupone persistencia de `/content/audit_ae_z32/`. El notebook comprueba
existencia después del montaje y ofrece un ZIP descargable que contiene
**exactamente los tres logs y metadata.json**, sin pesos. `stage_log_bundle`
rechaza destinos existentes o dentro del run histórico y compara hashes de
fuentes/copias/ZIP antes y después. Un archivo que cambia invalida el bundle.

Localmente solo se encontraron histories smoke de épocas 1–2. No se accedió a
Drive desde este entorno Windows y no hay copias locales de los logs de
producción 8–15. El informe JSON local usa exclusivamente la transcripción
histórica ya disponible y la identifica como tal. Los registros de gradientes,
eventos y metadata de producción quedan **pendientes de lectura en Colab**;
no se inventan cuantiles, conteos ni marcas de tiempo.

La transcripción histórica permite describir esta ventana, con valores
redondeados y sin rellenar épocas ausentes:

| Época | Train, nats/secuencia | Validación histórica, nats/secuencia |
| ---: | ---: | ---: |
| 8 | 164.70 | 168.67 |
| 9 | Sin dato local | Sin dato local |
| 10 | Sin dato local | Sin dato local |
| 11 | 159.07 | 172.69 |
| 12 | 709.80 | 1212.92 |
| 13 | 1043.41 | 336.01 |
| 14 | 882.82 | 1131.97 |
| 15 | 1490.27 | 1333.13 |

Entre 11 y 12 train aumenta unas **4.46 veces** y val unas **7.02 veces**.
Ambas empeoran y luego oscilan; el sobreajuste ordinario, por sí solo, explica
mal ese salto conjunto. Las medias de época no indican si el fallo ocurrió
al principio o al final de época 12, ni si un cambio de estado precedió al salto.

El analizador nuevo `src.deterioration_audit` examina:

- History: orden, duplicados, datos ausentes, cambios adyacentes y referencia
  de época 8; no interpola ni llama "causal" a un salto descriptivo.
- Gradientes por **sesión y época**: media/máximo preclip, máximo postclip,
  fracción recortada entre filas registradas, escalas y colas. Cuantiles exactos
  hasta 2.048 normas; después, muestra reservoir determinista acotada, etiquetada
  como aproximada. Filas muestreadas no equivalen a todos los lotes de una época.
- Eventos: distingue clipping de gradientes **finitos**, NaN/Inf y rechazos.
  El nombre `nonfinite_events.jsonl` no significa que todas sus filas sean NaN.
  Archivo ausente no equivale a cero eventos; archivo vacío tampoco prueba
  cobertura completa de la instrumentación.
- Alineación: solo cruza índices de lote cuando sesión/época están registradas;
  history sin sesión solo admite coincidencia por época. No une intentos
  repetidos ni establece precedencia causal por el nombre de un archivo.
- Metadata: parámetros, clipping, backend, reanudaciones y versiones, con
  procedencia y hashes; nunca lo usa como sustituto de una traza por lote.

## Revisión de la implementación original

Se revisaron `autoencoder.py`, `_epoch` y `run_experiment` en `train_ae.py` y
las pruebas existentes. Estos archivos se mantienen sin cambios en fase 4.

**Encoder y decoder.** Embedding→GRU→proyección lineal z32; el decoder recibe z
mediante `tanh(latent_to_hidden(z))` y como entrada repetida en los 80 pasos.
No recibe x ni teacher forcing. El latente lineal no está acotado; tanh en h0
solo limita una de las dos rutas. Estados ocultos acotados no garantizan
Jacobianos pequeños. Una fracción alta de |h|≥0.99 es un proxy de saturación
del estado, no una medida de probabilidades de gates GRU.

**Objetivo.** CE categórica contra x, logits `(B,80,76)`, suma sobre T y media
por B. No hay softmax previo a CE, broadcasting indebido ni objetivo y. UNK=0
participa y no existe PAD. La derivada respecto a logits está acotada por 1/B;
eso no acota la derivada respecto a parámetros a través del sistema recurrente.

**Acumulación.** `optimizer.zero_grad(set_to_none=True)` precede cada forward
de entrenamiento; se ejecuta un backward y un `optimizer.step()` por lote
aceptado. Los lotes rechazados no llegan a Adam. No se encontró evidencia en
el código de acumulación accidental ni de un hidden transportado entre lotes.
Sin logs/código exactos de cada sesión histórica no se certifica retrospectivamente
que todas las ejecuciones usaran idéntica versión.

**Clipping FP64.** La norma convierte antes de elevar al cuadrado/reducir; el
escalado también usa FP64, vuelve al dtype original y verifica finitud y norma
posterior ≤5·(1+1e−6). Gradientes enormes pero finitos no se confunden con una
reducción FP32 desbordada. La corrección de clipping no demuestra que desaparezca
el problema dinámico; tampoco hay evidencia de que causara el deterioro.

**Adam.** Mantiene momentos por coordenada: aproximadamente
`Δθ = −lr * m_hat / (sqrt(v_hat) + eps)`. Recortar la norma global del gradiente
a 5 no equivale a imponer una cota lr·5 a la actualización de parámetros. Se
necesitan momentos, escalas y Δθ/θ medidos, no deducidos de la frecuencia de
clipping. El trainer original verifica pérdida/gradientes antes de Adam y
comprueba pesos/momentos después de la época train+val. El control nuevo añade
verificación **después de cada paso**, reduciendo la latencia de detección.
Esta diferencia de observación no demuestra corrupción histórica de Adam.

**Contrato de checkpoints.** El trainer solo reanuda `last_resumable`. Un best
es de inferencia; durante recuperación puede componerse con pesos de best y
estado de optimizador de last. No se debe asumir alineación temporal de esos
momentos con best8. El rechazo de best como fuente de reanudación es deliberado;
no hay evidencia de que ese estado incongruente se utilizara en el entrenamiento
histórico. El nuevo runner no restaura Adam desde best/last de fases anteriores.

## Hipótesis y pruebas que las distinguen

| Hipótesis | Evidencia actual | Falta para establecer mecanismo |
| --- | --- | --- |
| Pérdida de reconstrucción específica y dependencia útil | Confirmada por endpoints pareados 8/30 | Trayectoria por updates para localizar inicio |
| Crecimiento de z y saturación recurrente contribuyen | Plausibles por rutas del modelo; correlaciones históricas comunicadas | z/estados/probes antes, durante y después del primer salto; distinguir precedencia de consecuencia |
| Dinámica de Adam deteriora las actualizaciones | Posible con momentos históricos y clipping frecuente | Δθ/θ y momentos por paso; replay pareado conservando datos/RNG |
| Gradientes recurrentes mal condicionados | Compatible con gradientes enormes comunicados y oscilación de pérdidas | Normas por bloque/VJP en checkpoints cercanos; espectros de pesos no bastan para el Jacobiano efectivo |
| Acumulación accidental | No respaldada por el trainer revisado | Comprobar reset por lote y versión histórica exacta |
| NaN/Inf inicia la crisis | Sin establecer: los logs aún no están disponibles localmente | Clasificar eventos reales y comprobar cobertura; gradientes grandes finitos no son NaN |
| Sobreajuste por sí solo | Empeoramiento de val antes del salto compatible; salto conjunto train/val no lo explica por sí solo | Separar generalización de inestabilidad de optimización |
| Error de objetivo/máscara/arquitectura | No se encontró en código/tests | Mantener regresiones; no cambiar arquitectura para eludir diagnóstico |

No se establece una causa raíz por descarte. La pérdida de dependencia es un
resultado a explicar; no es por sí misma el mecanismo que la inició.

## Ventana controlada y checkpoints protegidos

`src.deterioration_control` carga el AE original sin tanh latente. Cada salida
requiere un directorio inexistente. Registra fingerprints del baseline train
completo y de los prefijos train/val; la reserva train es fija de 256·batch
ejemplos, con una única permutación local. Este muestreo difiere del recorrido
histórico por épocas; se declara y no se vende como reproducción exacta.

Modos explícitos:

- `fresh`: inicialización nueva; comprueba instrumentación, no replica best8.
- `weights_only`: copia pesos históricos y crea Adam nuevo. No reproduce el
  optimizador, RNG ni cursor históricos; sus mejoras no prueban causalidad Adam.
- `continue`: solo checkpoint nuevo `phase4_periodic_resumable`; conserva
  pesos, Adam, RNG, cursor, orden, configuración y controles compatibles.
- `reset_adam`: solo ese formato nuevo, conservando pesos/RNG/cursor/datos/
  controles con Adam nuevo. Comparado con `continue` desde la misma fuente y
  presupuesto, aísla el reinicio del optimizador en los mismos lotes siguientes.
  Las ramas se ejecutan individualmente, nunca como una matriz automática.

Cada snapshot guarda pesos, Adam, RNG, cursor, contratos de datos, métricas y
estado de los controles. Se escribe mediante temporal y commit atómico; un
temporal cortado no es un checkpoint confirmado. Los archivos `update_*.pt`
son inmutables. Solo cambian punteros dentro de la carpeta nueva:
`latest_checkpoint.json` y `best_prefix.json`. Un estado alertado no sustituye
al mejor estado sano aunque presente menor CE. El checkpoint inicial también
se conserva; una parada no destruye estados anteriores ni modifica best8.
Para continuar se conserva la carpeta de la ventana anterior: un archivo
periódico descargado por sí solo puede referenciar un best anterior. El runner
verifica existencia, rol, fingerprint, métricas y hash de esa referencia y
rechaza continuar si falta o no coincide.

Por actualización se registra reset de gradientes, pérdida, normas preclip por
bloque y globales, norma/escalado postclip, magnitud de z, pesos antes/después,
Δθ real global/por bloque, Δθ/θ y resúmenes de pasos/momentos Adam. No se afirma
que una razón alta aislada sea una causa. En evaluación se añade saturación
encoder/decoder/h0 y magnitudes de z/logits sobre el primer batch diagnóstico
de hasta 32; CE/swap/TV se calculan sobre el prefijo completo de 128. Se conserva
el detalle por posición del módulo de fase 3.

La referencia de swap solo se activa tras observar utilidad previa ≥1 nat por
secuencia; evita llamar "colapso aprendido" a fluctuaciones del modelo aleatorio.
Volver a observar el checkpoint fuente al continuar no cuenta como un nuevo
intervalo de paciencia. Un checkpoint ya detenido no habilita continuación
automática para saltarse las protecciones.

## Criterios objetivos de alerta y parada

| Condición | Acción |
| --- | --- |
| NaN/Inf de pérdida/z/gradiente o norma/escalado inválido | Parar antes de Adam; no omitir lote silenciosamente |
| Estado/momento no finito después de Adam | Parar inmediatamente; conservar último checkpoint anterior válido y distinguir fallo posterior al paso |
| Norma raw >1e8, o magnitud absoluta de z >100 | Parar la rama antes de nueva actualización |
| CE diagnóstica ≥2 veces la mejor referencia sana | Parada inmediata |
| CE diagnóstica ≥1.20 veces la mejor referencia durante 2 evaluaciones consecutivas | Parar por deterioro sostenido |
| Daño swap ≤50 % de su mejor referencia informativa previa durante 2 evaluaciones consecutivas | Parar por pérdida sostenida de utilidad |
| Primera evaluación alertada | No promover como mejor; conservar snapshot y observar hasta paciencia o severidad |
| 100 % clipping, sin otro criterio | Registrar; no parar solo por esa frecuencia |
| Tiempo/datos/budget agotado | Terminar como acotado o incompleto; no declarar estabilidad de largo plazo |

Con los endpoints confirmados, época30 supera 1.20·CE8 y conserva 1.01 % del daño
swap8. Dos observaciones de ese tipo activarían los controles sostenidos; esto
es una comprobación aritmética del criterio, **no** evidencia de dos evaluaciones
históricas consecutivas ni de que se habría detenido exactamente en época12.
La comprobación severa de train añade una guarda contra valores mayores que
4 veces la mejor CE diagnóstica, con su alcance distinto registrado.

Estos umbrales son conservadores y prespecificados para el diagnóstico; no se
calibran con test ni se presentan como óptimos. Pueden requerir evaluación
posterior de falsos positivos sobre muestras/semillas predefinidas. Al saltar
un criterio se preserva evidencia, no se baja LR ni se reinicia Adam en secreto.

`--full-validation` calcula CE de todo val en streaming al inicio y al final de
una ventana completada; una rama detenida no ejecuta la evaluación final,
con ponderación correcta del último lote parcial. Se registra por separado;
swap y paradas siguen usando el prefijo. No hay selección automática de un
"mejor modelo global" usando solo 128 ejemplos. Tampoco hay intervalos IID:
las ventanas pueden solaparse y concentrar hablantes.

## Reproducción, validación y entregables

El notebook `notebooks/Workshop_4_Deterioration_Colab.ipynb` comienza con el plan
y los controles. Verifica las rutas confirmadas, crea el ZIP de logs y permite
auditoría de lectura; `RUN_WINDOW=False` mantiene entrenamiento desactivado.
No instala paquetes ni descarga Shakespeare. El ZIP de proyecto nuevo es
`exports/workshop4_colab_deterioration_phase4.zip`; no incluye datos ni pesos.

Plan sin entrenar, desde Workshop_4:

```bash
python -m src.deterioration_control \
  --config config/deterioration_control.json \
  --processed-dir /ruta/processed --device cuda \
  --source-checkpoint /ruta/best.pt --initialization weights_only \
  --output-dir /ruta/nueva/ventana
```

`--execute` habilita solo esa ventana acotada. CLI supervisa un hijo con timeout
max_seconds+20; Colab añade 650 s. Las guardas blandas operan entre operaciones;
un corte duro puede dejar temporales y resúmenes parciales. No tratar ausencia
de resumen final como ejecución completada. No hay `--all`, épocas ni VAE.

Auditoría de copias, sin pasos de optimizador:

```bash
python -m src.deterioration_audit \
  --history /copia/history.csv --gradient-norms /copia/gradient_norms.csv \
  --events /copia/nonfinite_events.jsonl --metadata /copia/metadata.json \
  --start-epoch 8 --end-epoch 15 --output-dir /ruta/nueva/auditoria --execute
```

La evidencia y validación locales de esta entrega están en
`audit/deterioration_phase4/`. Los tests usan fixtures y pocos pasos sintéticos;
no son una reproducción del deterioro en Shakespeare. Comprueban procedencia,
copias/ZIP inmutables, ausencia vs cero, sesiones, cuantiles, umbrales, mejor
checkpoint, finitud, reset de gradientes y reanudación/control del optimizador.
Los resultados finales y hashes se registran en `validation.json`.
Resultado local: **149 pruebas aprobadas, cero errores y cero omisiones**,
incluidas 35 nuevas de esta fase (16 de auditoría y 19 del controlador).
Los 105 archivos protegidos conservaron sus hashes. Se validaron las celdas
del notebook sin ejecutarlas y el CLI en modo plan sin salidas ni entrenamiento.
La validación del ZIP y su extracción se registra en `package_validation.json`.
No se realizó entrenamiento completo, commit ni push.

## Limitaciones y continuidad hacia anomalías

La fase permite observar y acotar daño en una rama nueva; no demuestra que la
causa histórica esté identificada ni que el deterioro se haya prevenido en
producción. Faltan los logs de Drive y estados completos cercanos al primer salto.
Una ventana corta sobre prefijos no valida 30 épocas, otros hablantes, AE z64
ni la estabilidad variacional. Una comparación con Adam reiniciado solo adquiere
valor causal acotado si conserva pesos, lotes, orden, RNG y presupuesto.

El objetivo final de anomalías con AE/VAE permanece. Antes de avanzar hacen
falta estabilidad y reconstrucción útil reproducibles; después, experimentos
normal-only, calibración de umbrales con datos separados y evaluación final
sin usar test para seleccionar. Para VAE habrá que revisar reconstrucción y KL
por separado. Ninguna de esas ejecuciones se activa desde esta fase.
