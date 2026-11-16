# Workshop 4 - Autoencoders y VAE

## Modelos normal-only restantes: AE z64 y VAE

[REMAINING_NORMAL_MODELS.md](REMAINING_NORMAL_MODELS.md) incorpora el AE z32 ya
terminado y prepara cuatro experimentos independientes. Ejecutar primero
`notebooks/Workshop_4_AE_z64_NormalOnly_Colab.ipynb`; los VAE z32 β=0.5/1/4 tienen
notebooks separados y desactivados. El ZIP nuevo es
`exports/workshop4_colab_ae64_vae_normal_only.zip`.

La comparación reutiliza los IDs/hashes del protocolo de desarrollo confirmado
(7.640 filas). AE z32 solo se evalúa; test permanece cerrado y no hay ejecución
automática de la matriz. Cada modelo parte de cero y guarda su carpeta en Drive.

## Primer AE z32 normal-only

[NORMAL_ONLY_ANOMALIES_PLAN.md](NORMAL_ONLY_ANOMALIES_PLAN.md) prepara un único
entrenamiento desde cero con los 562.253 normales de train, early stopping,
clipping FP64 y mejor checkpoint protegido. Selección, calibración y evaluación
usan tres grupos val disjuntos; se excluyen entradas idénticas a train de las
vistas held-out. Test requiere modelo, scores y umbrales sellados.

Notebook T4: `notebooks/Workshop_4_NormalOnly_AE_Colab.ipynb`. Paquete nuevo:
`exports/workshop4_colab_normal_only_ae.zip`. Incluye métricas completas y una
comparación pareada con best8 congelado. No continúa Adam histórico ni activa
AE z64 o VAE; la ejecución se habilita explícitamente en el notebook.

## Best8: restauración verificable y anomalías congeladas

[BEST8_RESTORE_AND_ANOMALIES.md](BEST8_RESTORE_AND_ANOMALIES.md) separa la
restauración mecánica de Adam de una continuación histórica fiel. El archivo
real de Drive se inspecciona sin updates; `continue` de Fase 4 conserva el estado
de sus ventanas, no el recorrido original por épocas. No se lanza otra ventana.

El notebook `notebooks/Workshop_4_Best8_Anomalies_Colab.ipynb`, en el ZIP nuevo
`exports/workshop4_colab_best8_anomalies.zip`, prepara scores MSE categórica/NLL,
calibración con normales de val y evaluación mixta sin restaurar Adam. El best8
general es un detector exploratorio: no sustituye el entrenamiento normal-only.
Test y evaluación están desactivados por defecto; no hay entrenamiento.

## Fase 4: diagnóstico del deterioro histórico

[AUDIT_REPORT_PHASE4.md](AUDIT_REPORT_PHASE4.md) incorpora la comparación
confirmada de best8/last30 y separa los resultados sobre 128 secuencias de la
validación completa. Prepara auditoría de logs 8–15 y ventanas cortas del AE
original, con checkpoints inmutables, mejor estado protegido y paradas por
deterioro. La causa raíz sigue sin establecerse.

Notebook: `notebooks/Workshop_4_Deterioration_Colab.ipynb`. ZIP nuevo:
`exports/workshop4_colab_deterioration_phase4.zip`. Incluye las rutas de Drive
confirmadas y una celda para descargar los tres logs con metadata.json, sin
pesos. RUN_WINDOW=False mantiene entrenamiento desactivado. No inicia AE z64,
VAE ni la matriz completa. Las fases anteriores conservan sus resultados.

## Fase 3: dependencia informativa del latente

[LATENT_INFORMATION_REPORT.md](LATENT_INFORMATION_REPORT.md) presenta los
diagnósticos nuevos: baselines ajustados con todo train, intervenciones de z,
comparaciones por posición y una prueba corta identity/tanh. En las 128 ventanas
val evaluadas al paso 24, intercambiar z apenas afecta la reconstrucción y ambos
AE pierden contra el baseline. La sensibilidad relativa histórica no basta para
demostrar pérdida de información previamente aprendida.

Usar `notebooks/Workshop_4_Latent_Information_Colab.ipynb` del ZIP nuevo
`exports/workshop4_colab_latent_phase3.zip`. Solo llama a
`src.latent_diagnostic_runner`; viene con ejecución y prueba corta desactivadas.
Los checkpoints se evalúan en lectura; cada salida exige una carpeta nueva.
No inicia AE z64, VAE, la matriz completa ni entrenamientos de 30 épocas.
Evidencia local: `audit/latent_information_phase3/`.

Fuente académica: `Workshop_4.pdf`. Se conservan los archivos de Fases 1 y 2.
Esta entrega prepara entrenamiento y recuperación; no contiene experimentos
completos ni resultados académicos de reconstrucción o anomalías.

## Pilotos controlados de estabilidad

[STABILITY_PLAN.md](STABILITY_PLAN.md) prepara la siguiente decisión experimental
tras finalizar ae_z32. Incluye referencia, batches 256/512, LR menor y tanh
opcional, cambiando un solo factor. `src.stability_experiments` compara por
ejemplos y por updates, con límites de trabajo/tiempo y salidas nuevas.
`config/stability_experiments.yaml` y el nuevo notebook
`notebooks/Workshop_4_Stability_Colab.ipynb` mantienen ejecución deshabilitada.
Solo --execute/EXECUTE=True inicia los pilotos; best.pt se evalúa sin Adam.
La opción tanh cambia la arquitectura y no se aplica a checkpoints históricos
ni a VAE. Una prueba corta aprobada no demuestra estabilidad de 30 épocas.

## Auditoría posterior de convergencia y rendimiento

El [informe técnico](AUDIT_REPORT.md) analiza las pérdidas suministradas hasta
época 26, separa evidencia de hipótesis y documenta los cambios y límites.
**No actualizar ni ejecutar este controlador mientras siga activo ae_z32 en
Colab.** El código se preparó localmente para una carpeta y salida nuevas cuando
termine ese proceso. No hay una causa primaria confirmada sin sus checkpoints.
Paquete actualizado: `exports/workshop4_colab_audited.zip`; conserva los ZIP
anteriores como referencia del entrenamiento histórico.

Herramientas nuevas: `src.audit_model` (copias de best/last, sin optimizer.step),
`src.audit_logs` (copias de CSV/journal/metadata), `src.benchmark_ae` (trials
aislados de pocos pasos) y profiling opt-in con `--profile-batches N`.
`--tensor-loader`, `--prefetch-batches N` y `--pin-memory` son opciones
explícitas de transporte; no se activan en continuaciones existentes.
`--gradient-log-every N` muestrea filas ordinarias sin omitir incidencias ni
alterar los agregados por época. Conserva FP64 y todas las protecciones.

`--recommended` selecciona AE z32/z64 y VAE z32 con beta 0.5/1/4; las doce
definiciones siguen disponibles mediante `--all`, que es una elección explícita.
El panel Colab propone las cinco y mantiene RUN_TRAINING=False. Son experimentos
de representación; el entrenamiento normal-only y el umbral val siguen siendo
tareas separadas de anomalías.

La evidencia CPU medida está en `audit/evidence/`; no se ha medido una mejora
end-to-end en Tesla T4. Los comandos precisos para benchmark y actualización
segura de Colab están en el informe. No cambiar pérdidas, LR, batch ni
arquitectura en un entrenamiento existente para aplicar estas opciones.

## Datos y tokens

Se reutilizan sin cambios los Parquet, vocabulario y particiones de Workshop 1.
Validación completa anterior: raw 4.226.158 filas, train 2.997.217, val 583.098,
test 644.398; ventanas de 80 caracteres y 76 tokens. El entrenamiento solo abre
**train y val**: no descarga datos ni usa test para escoger hiperparámetros.

El vocabulario contiene `<UNK>` en 0 y **no tiene PAD**. Todas las posiciones
contribuyen a la pérdida; no hay máscara ni `ignore_index=0`. El cargador valida
hablantes, longitudes y `y_id`. Reconstrucción usa x como objetivo, no y.

## Arquitecturas y objetivo

`src/autoencoder.py`: embedding -> GRU -> bottleneck -> GRU condicionada solo
en z -> logits `(B,80,76)`. VAE incluye mu/logvar, reparametrización y KL respecto
a N(0,I). No se pasa x al decoder ni se usa teacher forcing.

Reconstrucción: entropía cruzada categórica sumada por secuencia. KL se suma
por dimensiones latentes; después se promedian ambas por ejemplos. Unidades:
**nats/secuencia**; dividir reconstrucción por 80 da nats/carácter. AE registra
KL=0; VAE total = reconstrucción + beta*KL. Validación VAE utiliza una muestra
posterior por ejemplo con semilla fija, aislada del RNG de entrenamiento: una
estimación Monte Carlo, no pérdida en mu. Para reconstrucciones/visualizaciones
posteriores deterministas se usará mu. El MSE del PDF se calculará en evaluación
sobre probabilidades frente a one-hot, nunca sobre IDs ordinales.

## Configuración y CLI

`config/experiments.yaml` contiene JSON, válido como YAML 1.2: no requiere
PyYAML. Sintaxis YAML plena se admite si se instala PyYAML opcional. Hay 12 runs:
AE con z=32/64/128 y VAE con esos tamaños cruzados con beta=0.5/1/4. Defaults:
Adam, 30 épocas, batch 512, lr 0.001, seed 42, embedding 32, hidden 128,
una capa recurrente, dropout 0 y clipping de gradiente a norma 5.

Desde Workshop_4 (Linux/Colab: sustituir el ejecutable por `python`):

```powershell
# Solo validar el plan y datos, sin entrenar ni escribir resultados:
& ../.venv/Scripts/python.exe -B -m src.train_ae --experiment ae_z32 --dry-run
# Solo cuando se autorice entrenamiento completo:
& ../.venv/Scripts/python.exe -B -m src.train_ae --experiment ae_z32 --device auto --epochs 30
& ../.venv/Scripts/python.exe -B -m src.train_ae --all --device cuda --output-dir runs/workshop4
# Continuar una grilla interrumpida:
& ../.venv/Scripts/python.exe -B -m src.train_ae --all --device auto --resume auto
```

Opciones editables sin cambiar scripts: `--epochs` (30-50), `--batch-size`,
`--learning-rate`, `--latent-dim`, `--beta`, `--seed`, `--embedding-dim`,
`--hidden-size`, `--num-layers`, `--dropout`, `--grad-clip`. `--experiment` se
puede repetir; `--all` ejecuta secuencialmente. Overrides afectan todos los
seleccionados y conservan sus nombres; valores efectivos van a metadata.json.
Usar un nuevo nombre/directorio si cambia la configuración. Para recorrer el
grid original conservar los valores individuales de z/beta sin overrides.

Rutas: `--data-config`, `--processed-dir`, `--vocab-path`, `--splits-path`.
Defaults se resuelven desde Workshop_4; rutas explícitas desde el directorio
actual. Se usa pathlib en Windows/Linux. No hacen falta raw/test para entrenar.

## Resultados y recuperación

La protección numérica y el procedimiento para continuar un checkpoint legado
sin sobrescribir su directorio original se describen en
[NUMERICAL_STABILITY.md](NUMERICAL_STABILITY.md). Los nuevos logs incluyen
incidencias y conteos explícitos de lotes/ejemplos rechazados; los promedios
train se calculan solo sobre ejemplos aceptados. Se mantienen clipping estricto,
Adam, pérdidas e hiperparámetros originales.

El clipping global L2 calcula norma y producto de escalado en FP64, con umbral
5.0 y verificación posterior sobre los gradientes que recibirá Adam. Gradientes
finitos grandes se clippean y registran sin omitir el lote; NaN/Inf siguen
sujetos a rechazo y a los mismos límites. `gradient_norms.csv` guarda normas
previas/posteriores por lote y el journal señala eventos de gradientes grandes.

Por cada run en `OUTPUT/NOMBRE/` se guardan:

- `history.csv`: reconstrucción/KL/total de train/val, época, conteos, lr y tiempo.
  Promedios ponderados por el número real de ejemplos.
- `metadata.json`: parámetros efectivos, dispositivo/versiones, contrato y
  SHA-256 de train/val, reglas de tokens y protocolo de validación/selección.
- `checkpoints/best.pt`: mínimo **val_total dentro del run**.
- `checkpoints/last.pt`: estado de modelo, Adam, RNG Python/PyTorch CPU/CUDA,
  época, historial, mejor estado, vocabulario y contrato de datos.

No comparar val_total directamente entre betas: beta cambia la escala. Se
compararán reconstrucción/separabilidad en fases posteriores. Aún no se
publica un único ae_best.pt/vae_best.pt global.

Se escribe a temporales y luego os.replace. last.pt es la fuente de recuperación
de CSV/best ante escrituras interrumpidas. Solo **last.pt** permite reanudar;
best.pt es de inferencia. Una interrupción a mitad de época repite esa época
desde el último checkpoint completo. Sin ningún last.pt, usar nueva salida.

`--resume auto` continúa runs existentes e inicia los aún no empezados de una
grilla. Checkpoint explícito solo para un run. Rechaza cambios de parámetros,
vocabulario, splits o bytes Parquet. Rutas pueden cambiar al pasar a Drive.
Carga con weights_only=True y map_location=cpu antes de trasladar al dispositivo.
La igualdad bit a bit se prueba en CPU con el mismo entorno; hardware/versiones
distintos pueden cambiar los números. Se activan algoritmos deterministas; no
se oculta un kernel CUDA incompatible desactivando esa garantía.

Se mezclan ventanas acotadas de 16 batches con seed+época, sin cargar todos
los datos en RAM. train/val nunca se mezclan entre sí.

## Exportación a Colab sin Git

1. Desde Workshop_4 exportar el ZIP (solo stdlib):

   ```powershell
   & ../.venv/Scripts/python.exe -B -m src.colab_utils --export-zip exports/workshop4_colab.zip
   ```

   Incluye código/config/docs/notebook/tests de W4 y solo vocab/splits de W1.
   Excluye .git, .venv, Parquet, logs y checkpoints. Se entrega un ZIP preparado;
   regenerarlo después de cambiar archivos. Ninguna operación Git es necesaria.

2. Subir ZIP a Drive y conservar los Parquet recuperados en una carpeta como
   `MyDrive/Workshop_4/data/processed/`. Abrir `notebooks/Workshop_4_Colab.ipynb`
   en Colab y elegir runtime GPU. También admite upload del ZIP desde el navegador.

3. Montar Drive y editar campos de rutas. Se extrae el código en /content. El
   controlador lee train/val de Drive o los copia a /content con hashes para
   acelerar las lecturas; no modifica los originales ni utiliza test/raw.

4. Revisar dependencias y panel: épocas, batch, lr, seed, dispositivo, nombres o
   todos los runs. z=0 y beta vacío conservan el grid. Se mantiene el PyTorch de
   Colab; solo se instalan auxiliares ausentes, sin forzar una build CPU.

5. Revisar el dry run antes de activar RUN_TRAINING (entregado en False). Los
   resultados se escriben directamente en OUTPUT_DIR de Drive. Otra sesión puede
   continuar con el mismo ZIP/datos/config/salida y RESUME activado.

6. La celda de curvas llama plot_history y guarda training_curves.png junto al
   CSV en Drive. Descargar desde Drive las carpetas y checkpoints deseados.

Notebook solo controla parámetros y llamadas. Transporte y curvas están en
src/colab_utils.py; entrenamiento en src/train_ae.py. El notebook fue validado
como JSON/AST, sin ejecutarse en Colab ni GPU. El entorno local del smoke es
CPU; matplotlib no se instaló localmente. Las curvas se comprobaron mediante
lectura de CSV y un backend simulado, no mediante renderizado local.

## Pruebas pequeñas y smoke

```powershell
& ../.venv/Scripts/python.exe -B -m unittest discover -s tests -p test_train_ae.py -v
& ../.venv/Scripts/python.exe -B -m unittest discover -s tests -p test_colab_utils.py -v
# Smoke real: máximo 2 épocas x 2 batches por partición, no entrenamiento completo.
& ../.venv/Scripts/python.exe -B -m src.train_ae --experiment vae_z32_b1 --smoke-test --epochs 2 --batch-size 2 --hidden-size 8 --embedding-dim 4 --max-train-batches 1 --max-val-batches 1 --stop-after-epoch 1 --device cpu --output-dir runs/workshop4/_smoke_manual
# Repetir sin --stop-after-epoch y con --resume auto para completar la recuperación.
```

Las pruebas verifican AE/VAE mínimos, recuperación VAE igual a ejecución continua
(pesos, Adam y métricas salvo tiempo), rechazo de configuración/datos, reparación
de CSV/best y filtro normal. El smoke real usa dos ejemplos de train/val por época,
con logs separados en runs/workshop4/_smoke_real. No son métricas académicas ni
sustituyen el requisito de 30-50 épocas.

Verificación local de esta entrega: 12 pruebas numéricas de Fase 2, 6 de
entrenamiento/recuperación y 6 de transporte/curvas aprobadas. Se validaron
además el grid/CLI/notebook y el smoke real AE/VAE con reanudación hasta época 2.

## Fases posteriores

El grid es para representación. Anomalías requiere runs separados **solo normales**:
otro nombre YAML y normal_class definido a partir de train. El filtro usa el y
existente sin cambiar splits; umbral en val y evaluación mixta posterior. No se
infiere normalidad desde test. Siguen pendientes evaluación/anomalías/AUC-ROC,
comparación supervisada congelada, t-SNE/UMAP/silhouette e informe PDF.
