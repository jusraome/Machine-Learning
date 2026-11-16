# Modelos normal-only restantes: AE z64 y VAE z32

Fecha: 16 de noviembre de 2026.

## Estado confirmado y siguiente ejecución

El AE z32 normal-only ya terminó en T4 con 562.253 secuencias normales, mejor
checkpoint en época 14/update 60.928 y finalización
`completed / early_stopping_normal_selection`. Su desarrollo terminó sobre
7.640 secuencias. Test sigue cerrado. Estos datos fueron confirmados por el
usuario; no se reentrena ese modelo ni se atribuye una reevaluación local nueva.

| Modelo confirmado, mismo desarrollo | ROC-AUC NLL | ROC-AUC MSE categórica |
| --- | ---: | ---: |
| AE z32 normal-only | 0.535611 | 0.527281 |
| Best8 histórico exploratorio | 0.502575 | 0.507950 |
| Diferencia descriptiva | +0.033036 | +0.019331 |

Hay mejora descriptiva respecto a best8, pero la discriminación sigue siendo
modesta. No hay una estimación de incertidumbre, una repetición con otras
semillas ni resultados de test que justifiquen una conclusión más fuerte.
No se cambia la etiqueta o se invierte un score a partir de estos resultados.

La siguiente ejecución preparada es **AE z64**, individual y desde cero.
Después quedan tres VAE z32 separados con β=0.5, 1 y 4. Cada notebook tiene
`RUN_TRAINING=False`; no hay ejecución automática de los cuatro, ni un CLI
`--all`, ni reanudación de checkpoints históricos. El AE z32 está excluido del
nuevo runner de entrenamiento.

## Revisión breve del protocolo antes de usar GPU

Se reutilizó el inventario completo anterior y se comprobaron los SHA de
train/val. Los archivos coinciden, por lo que no hizo falta repetir una auditoría
de Adam ni volver a puntuar modelos históricos para preparar los experimentos.

- Normal sigue siendo la clase modal de **y, siguiente carácter, en train**:
  espacio, ID1. Train tiene 562.253 normales de 2.997.217 filas.
- Anomalía positiva=1 cuando y es otra clase. Las anomalías agregadas suponen
  81.2408% de train y 81.2730% del val completo. La clase normal es modal, aunque
  tenga menos del 50% de filas. No son anotaciones semánticas de novedad.
- El modelo reconstruye x; y solo filtra datos normales y etiqueta las métricas.
  NLL y MSE más altos indican anomalía. UNK se conserva; no se introduce PAD.
- AP debe contrastarse con la prevalencia de la población efectivamente
  evaluada. AP alta, por sí sola, no prueba discriminación. AP y PR-AUC
  trapezoidal se reportan separadas, con empates y convenciones documentados.
- Se conservan los tres grupos val por hash/seed 42:57 hablantes para selección,
  56 para calibración y 56 para evaluación. Solo normales de selección eligen
  checkpoint/early stopping; solo normales de calibración fijan umbrales.
- Continúan las cuarentenas de x idénticos a train normal. Si se compara best8,
  también se comprueban coincidencias con todas las clases de train. Las vistas
  cambian; los archivos originales y todas las filas normales de train permanecen.

Limitaciones previas al gasto: las etiquetas proxy pueden ser poco separables
por reconstrucción; las ventanas pueden solaparse; separar hablantes no demuestra
independencia ni ausencia de similitud semántica. Las dimensiones latentes y β
pueden producir diferencias de capacidad/regularización sin que la AUC mejore.
No se promete que un modelo más grande o un VAE resuelva esa limitación.

## Cohorte exacta de desarrollo y rutas confirmadas

Checkpoint AE z32 proporcionado por el usuario:

```text
/content/drive/MyDrive/Workshop_4/normal_only_ae/ae_z32_normal_only_20261116T134404_817936Z/best_inference.pt
```

Referencia proporcionada:

```text
/content/drive/MyDrive/Workshop_4/normal_only_ae/development_20261116T134404_817936Z/frozen_protocol.sha256
```

El JSON asociado esperado es `frozen_protocol.json` en esa misma carpeta. Los
notebooks verifican que ambos existan después de montar Drive, que el SHA sea
correcto, que el AE z32 corresponda a época 14/update 60.928 y que la referencia
contenga exactamente7.640 filas de evaluación. Las rutas se pueden corregir en
el panel si los archivos fueron movidos; el contenido y hash deben coincidir.
También resuelven train/val, sus roles y los IDs/hashes de la referencia en una
comprobación de lectura previa al entrenamiento, sin actualizar modelos. Una
referencia JSON válida no sustituye comprobar el contenido de los datos actuales.

`evaluate_normal_models.py` resuelve los IDs originales de calibration y
evaluation y verifica contenido, orden, roles, clase normal, contrato y guarda
de train. Reutiliza exactamente esas filas, sin muestrear ni rellenar huecos.
Un hash/ID ausente o una contradicción de cuarentena aborta la comparación;
no se modifica la cohorte para lograr el mismo tamaño por otra vía.

También verifica el SHA del AE z32 fijo y, si se incluye, best8 contra la
referencia. Cambiar un checkpoint por otro de igual forma no pasa inadvertido.
Los modelos nuevos reciben umbrales propios sobre los mismos normales de
calibración; las métricas se calculan sobre las mismas7.640 entradas mixtas.
La partición de selección normal se conserva como antes, sin usar los scores
de calibration/evaluation para escoger pesos.

Los archivos de Drive no están presentes en esta máquina. La verificación de
sus bytes se ejecutará al montar Drive; aquí se validaron herramientas, planes,
fixtures e invariancia de las entregas anteriores. No se fabrican filas o
métricas del protocolo real para sustituirlo.

## Experimentos independientes y arquitectura

| ID | Modelo | z | β | Parámetros con V=76 | Estado |
| --- | --- | ---: | ---: | ---: | --- |
| ae_z32 | AE original | 32 | No aplica | 145.004 | Ya entrenado; solo evaluación |
| ae_z64 | AE original | 64 | No aplica | 165.516 | Preparado para ejecutar primero |
| vae_z32_b0p5 | VAE existente | 32 | 0.5 | 149.132 | Preparado; ejecutar después individualmente |
| vae_z32_b1 | VAE existente | 32 | 1 | 149.132 | Preparado; ejecutar después individualmente |
| vae_z32_b4 | VAE existente | 32 | 4 | 149.132 | Preparado; ejecutar después individualmente |

AE z64 cambia únicamente el ancho solicitado del cuello de botella y las
matrices que conectan con él. Conserva embedding 32, hidden 128, una capa GRU,
dropout0 y la estructura original. Por eso aumentan los parámetros; no es una
comparación igualada en número de pesos.

Los VAE utilizan la clase variacional existente de `autoencoder.py`: encoder
GRU, cabezas μ/logvar, reparametrización y decoder original. No se modifica esa
clase. Los tres parten de pesos nuevos, Adam nuevo, la misma semilla y el mismo
orden de datos; solo cambia β. No hay tanh latente nuevo, annealing de β,
free-bits, teacher forcing ni conexiones adicionales.

Cada configuración está en `config/remaining_models/`. Los controles se heredan
del AE z32 normal-only: batch 128, Adam lr 0.0003, seed 42, hasta 30 épocas,
150.000 actualizaciones y 4 h por ejecución, con techo absoluto6 h. Train usa todas
las 562.253 filas normales, filtradas antes del shuffle/batching, incluido el
último lote parcial. No se vuelve a un piloto de prefijo reducido.

## Pérdidas, selección de checkpoints y efectos de β

AE z64 entrena con CE categórica de reconstrucción de x, suma temporal y media
por ejemplos. VAE entrena con:

`reconstrucción(z muestreado) + β · KL(q(z|x) || N(0,I))`.

Reconstrucción suma las 80 posiciones; KL suma las 32 dimensiones y ambos se
promedian por ejemplos. Se registran por separado reconstrucción, KL sin
ponderar, β·KL y objetivo total. Un valor menor de objetivo ponderado con β
distinto no basta para declarar un mejor modelo: su ponderación cambió.

Para mantener el criterio de selección común con AE z32, **todos eligen el
mejor checkpoint y aplican early stopping por reconstrucción normal de
model_selection**. En VAE esa reconstrucción se evalúa determinísticamente en
μ, con `sample=False`. Se registra también KL y `objective_at_mu = recon(μ)+βKL`.
Ese objetivo en μ **no es una estimación Monte Carlo de ELBO ni la
log-verosimilitud marginal**. El objetivo optimizado durante train sí usa
reparametrización muestreada. No se confunden ambos cálculos.

Se registran por VAE varianza/dispersión de μ, rango de logvar, KL por dimensión
y unidades activas (varianza de μ>0.01 en la muestra fijada), además de daño por
intercambiar μ, respuesta a cero y TV. Son diagnósticos dependientes de escala,
no estimadores de información mutua. KL pequeña y pocas unidades activas pueden
indicar poco uso posterior, pero también reflejar regularización fuerte; no
establecen por sí solas un fallo numérico ni una causa histórica.

Se espera que β cambie el compromiso reconstrucción/regularización, pero la
dirección y magnitud concretas deben medirse. La comparación final conserva
reconstrucción y KL por separado, con sus épocas/updates realmente seleccionados.

## Protecciones mantenidas

- Clipping global a5 con norma y escalado FP64, finitud previa y postcondición.
- Un reset de gradientes, un backward y un paso Adam por lote válido; pérdida,
  latente/posterior, gradientes, parámetros y momentos se verifican por paso.
- Guardas de norma raw>1e8 y magnitud latente>100; NaN/Inf o estado inválido
  detienen el experimento sin omitir silenciosamente datos.
- Deterioro severo de CE normal≥2×mejor referencia, deterioro sostenido≥20% en
  dos evaluaciones y pérdida sostenida de utilidad latente previamente
  informativa conservan el comportamiento del AE z32. Clipping frecuente por
  sí solo no detiene el entrenamiento.
- Early stopping: cinco evaluaciones sin mejora relevante de 0.1 nat/secuencia,
  después de tres épocas completas. El warmup no desactiva guardas numéricas.
- Evaluación/checkpoint cada 512 updates y al final de época. Checkpoints
  inmutables, inicial preservado y mejor estado sano protegido. Un modelo con
  cero actualizaciones aceptadas no se exporta como entrenado.
- La exportación `best_inference.pt` es para inferencia. Este runner no admite
  reanudación, checkpoints fuente ni matrices; no interpreta estados periódicos
  como recuperación exacta de un buffer interrumpido.

Las mismas guardas de utilidad pueden detener un VAE si pierde una dependencia
antes aprendida. Ese motivo debe leerse junto a KL/β y reconstrucción: no prueba
automáticamente que la dinámica variacional sea numéricamente inestable.
Una ejecución terminada o acotada tampoco demuestra estabilidad por 30 épocas.

## Scores y evaluación común sin test

Scores comunes y predeclarados:

1. `brier_mse`: MSE de probabilidades softmax frente a one-hot de x, media sobre
   posiciones y categorías; score principal del protocolo existente.
2. `nll_nats_per_sequence`: NLL reconstructiva de x, suma sobre posiciones.

Para VAE ambos se calculan en μ, de forma determinista. Además se registra KL
y se evalúan dos scores secundarios, nombrados por separado:

- `nll_plus_kl_nats_per_sequence`: NLL(μ)+KL, con coeficiente1 común a los VAE.
- `nll_plus_training_beta_kl_nats_per_sequence`: NLL(μ)+βKL, ligado al β de train.

No se suma KL a MSE, cuyas unidades/escala son diferentes. Los scores compuestos
en μ no se presentan como ELBO Monte Carlo. No se elige entre scores mirando
test ni se invierte la dirección para corregir una AUC inferior a0.5.

Cada score/modelo calibra solo con normales del grupo calibration, con alpha 0.05
y la misma regla de orden y excedencia estricta del AE z32. Se sellan modelos,
umbrales, scores e IDs antes del primer forward mixto. ROC-AUC, Average
Precision, PR-AUC trapezoidal, precision, recall, F1, FPR y confusión se registran
junto con prevalencia, curvas, cobertura y distribuciones. No se sustituyen
métricas indefinidas por números inventados.

El evaluador nuevo **no tiene modo test**. Los nuevos protocolos indican test
cerrado y no eligen un ganador automáticamente. El `Test ready: True` del AE z32
significaba que existía calibración suficiente; no era autorización para evaluar
test. La comparación de desarrollo permite decidir más adelante qué modelo y
score fijar antes de una única evaluación final en otra etapa.

## Notebooks, orden y rutas de salida

Paquete nuevo: `exports/workshop4_colab_ae64_vae_normal_only.zip`.

1. **Ahora:** `notebooks/Workshop_4_AE_z64_NormalOnly_Colab.ipynb`.
2. Después: `Workshop_4_VAE_z32_b0p5_NormalOnly_Colab.ipynb`.
3. Después: `Workshop_4_VAE_z32_b1_NormalOnly_Colab.ipynb`.
4. Después: `Workshop_4_VAE_z32_b4_NormalOnly_Colab.ipynb`.
5. Cuando existan los modelos: `Workshop_4_Normal_Models_Comparison_Colab.ipynb`,
   que solo evalúa y no contiene entrenamiento.

Cada notebook de train fija su ID y un solo JSON. Revisa la referencia, muestra
el plan y únicamente entrena al activar `RUN_TRAINING=True`. Abrir o ejecutar
las celdas con el valor por defecto no entrena. La comparación usa el AE z32
existente como referencia, el candidato nuevo y los otros modelos ya terminados
que se indiquen. Puede agregarse best8 solo como baseline exploratorio.

Los datos train/val existentes se copian o reutilizan localmente para E/S;
test no se copia. Las salidas van a carpetas nuevas bajo
`/content/drive/MyDrive/Workshop_4/remaining_normal_models/<ID>_<timestamp>/`.
Reabrir un notebook permite indicar `EXISTING_TRAINING_DIR` para evaluar un
modelo terminado sin volver a entrenarlo. No se escoge automáticamente la
carpeta más reciente ni se sobrescriben resultados previos.

CLI para preparar primero AE z64 (añadir `--execute` solo al iniciar esa corrida):

```bash
python -m src.train_remaining_normal \
  --config config/remaining_models/ae_z64.json \
  --processed-dir /ruta/processed --device cuda --output-dir /drive/nuevo/ae_z64
```

La comparación acepta únicamente checkpoints ya existentes, por ejemplo:

```bash
python -m src.evaluate_normal_models \
  --checkpoint ae_z32=/drive/ae_z32/best_inference.pt \
  --checkpoint ae_z64=/drive/ae_z64/best_inference.pt \
  --reference-protocol /drive/development/frozen_protocol.json \
  --processed-dir /ruta/processed --device cuda --output-dir /drive/nuevo/comparacion
```

Los tres VAE se ejecutan con sus JSON individuales; no se entrega un bucle de
entrenamientos. El CLI supervisa los plazos y el notebook añade un timeout
exterior. Un corte duro deja salidas parciales: revisar summary y checkpoints,
sin relanzamientos automáticos ni afirmaciones de épocas completadas.

## Verificaciones y límites de esta entrega

Las nuevas pruebas usan fixtures y entrenamientos sintéticos mínimos. Verifican
el rechazo de AE z32 en el runner, independencia de Adam, control por β,
reparametrización diferenciable, selección determinista en μ, FP64/guardas,
mejor protegido, roles/filtros normales y la reutilización exacta de cohortes.
La evaluación verifica fuentes/RNG y que no se entrene ni se lea test.

Evidencia: `audit/remaining_normal_models/`, con resultados confirmados del AE
z32 identificados como comunicados, revisión breve de protocolo, parámetros,
hashes y validación de pruebas/notebooks/ZIP. Aquí no se entrenó Shakespeare ni
se calcularon AUC nuevas de AE z64 o VAE. No se repitieron auditorías históricas
de Adam, no se modificaron modelos/datos previos y no se hizo commit ni push.

Validación final local: **262 pruebas aprobadas, cero errores y cero omisiones**,
incluidas27 nuevas (13 del trainer y 14 del evaluador). Los163 archivos protegidos
conservaron sus hashes. Se validaron los cinco notebooks sin ejecuciones ni
salidas, los cuatro flags de entrenamiento desactivados, los cinco planes CLI
sin efectos y el paquete/extracción. Evidencias: `validation.json` y
`package_validation.json` en la carpeta de esta entrega. No se entrenó AE z32,
AE z64 o VAE de producción ni se ejecutó evaluación de test localmente.
