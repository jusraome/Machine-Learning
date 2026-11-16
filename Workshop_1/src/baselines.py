"""baselines.py

Modelos base (Regresión Logística, Random Forest y Naive Bayes) para
predicción del siguiente carácter, sobre el dataset Shakespeare de LEAF ya
dividido en train/val/test por personaje (ver src/dataset.py), con logging
en 'runs/' y resumen en 'checkpoints/'.

Correcciones respecto a la versión anterior (todas afectan el rendimiento o
la validez de la evaluación):

 1. Muestreo ESTRATIFICADO de train. Antes era aleatorio simple: con 30k
    filas de 3M, las clases raras podían quedar fuera de la muestra por
    completo y el modelo nunca las veía.
 2. Más datos: 200k filas de train por defecto (era 30k) y evaluación sobre
    val/test COMPLETOS (antes también se muestreaban, así que las métricas
    tenían ruido de muestreo).
 3. Contexto de 10 caracteres (era 5). El EDA midió que con k=5 todavía
    queda el 37.6% de la incertidumbre y la curva seguía bajando.
 4. N-gramas de caracteres (2 a 4) sobre los últimos 12 caracteres como
    features adicionales para LR y NB. El one-hot posicional no puede
    representar pares como 'q'->'u' (100% en el EDA); los n-gramas sí.
    El vectorizador se ajusta SOLO con train: sin fuga de información.
 5. Dos baselines de referencia que faltaban (los propuestos en el README):
    - Clase mayoritaria (predecir siempre ' ': ~18.74% de accuracy).
    - Tabla de búsqueda n-gram con backoff k=3->2->1->mayoría, construida
      sobre TODO el train (es solo contar: no cuesta casi nada y es el
      "peso mínimo" que cualquier modelo debe superar).
 6. Protocolo de evaluación: la búsqueda de hiperparámetros del RF se hace
    sobre un subconjunto de train y el ganador se REAJUSTA con todo el
    train disponible. Las métricas de val de los modelos ajustados por
    val son optimistas por definición; las honestas son las de test.
 7. Métricas que faltaban: log-loss en BITS (comparable con la entropía
    H(y) = 4.5002 bits medida en el EDA) y top-3 accuracy.
 8. El AUC-ROC macro (One-vs-Rest) se calcula sobre un subconjunto fijo
    de como máximo 50k filas: es una métrica de ordenamiento y 50k filas
    la estiman igual de bien, pero cuesta 10x menos tiempo.

Uso:

    python -m src.baselines                       # valores por defecto
    python -m src.baselines --train-rows 50000    # menos filas, más rápido
    python -m src.baselines --context 5           # contexto más corto

Requiere haber ejecutado antes: python -m src.download y
python -m src.dataset --prepare (los parquet de data/processed/).
"""

from __future__ import annotations

import argparse
import gc
import time
from datetime import datetime
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OneHotEncoder

# Logging en TensorBoard. Se prefiere torch; si no está, tensorboard puro.
# OJO: la API de tensorboard.summary.Writer NO es la de torch (constructor con
# `output` en vez de `log_dir`, add_scalar con `step` en vez de `global_step`,
# y sin add_figure): sin este adaptador, instalar tensorboard - que es lo que
# sugería el mensaje de advertencia - hacía crashear el script apenas creaba
# el writer.
class _TorchLikeWriter:
  """Expone la API de torch.utils.tensorboard.SummaryWriter sobre Writer."""

  def __init__(self, log_dir: str):
    from tensorboard.summary import Writer

    self._writer = Writer(output=log_dir)
    self._figure_warning = False

  def add_scalar(self, tag, value, global_step=None, wall_time=None):
    self._writer.add_scalar(tag, value, 0 if global_step is None else global_step)

  def add_figure(self, tag, figure, global_step=None, close=None):
    # Writer no soporta figuras: quedan solo las copias PNG del run.
    if not self._figure_warning:
      print(
          "  (nota: tensorboard puro no soporta add_figure; las matrices de"
          " confusión quedan como PNG en el directorio del run)"
      )
      self._figure_warning = True

  def close(self):
    self._writer.close()


def make_writer(log_dir: Path):
  """SummaryWriter de torch, o el adaptador de tensorboard puro, o None."""
  try:
    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter(log_dir=str(log_dir))
  except ImportError:
    pass
  try:
    return _TorchLikeWriter(str(log_dir))
  except ImportError:
    print(
        "Advertencia: No se encontró 'tensorboard' ni 'torch'. Ejecuta: pip"
        " install tensorboard"
    )
    return None


# -------------------------------------------------------------------------
# 1. Configuración de Entorno y Rutas
# -------------------------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent  # .../Workshop_1/src
PROJECT_ROOT = SRC_DIR.parent  # .../Workshop_1

# Carpetas requeridas
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RUNS_DIR = PROJECT_ROOT / "runs"

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# Crear subcarpeta específica con timestamp para esta ejecución
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
CURRENT_RUN_DIR = RUNS_DIR / f"run_{RUN_TIMESTAMP}"
CURRENT_RUN_DIR.mkdir(parents=True, exist_ok=True)

# Ruta de destino del resumen CSV
csv_output_path = CHECKPOINTS_DIR / "baselines_summary.csv"

# Agregar rutas a sys.path
for path in (PROJECT_ROOT, SRC_DIR):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

try:
  from src.config import SEED, set_seeds
  from src.dataset import load_split, read_vocab

  set_seeds()
except ImportError:
  try:
    from config import SEED, set_seeds
    from dataset import load_split, read_vocab

    set_seeds()
  except ImportError as e:
    raise ImportError(
        f"No se pudo encontrar 'dataset.py' ni 'config.py'. "
        f"Verifica que existan en '{SRC_DIR}' o '{PROJECT_ROOT}'. Detalle: {e}"
    )

# -------------------------------------------------------------------------
# Parámetros globales (los tres primeros son CLI-overridables)
# -------------------------------------------------------------------------
CONTEXT_LENGTH = 10      # caracteres de contexto one-hot (era 5)
MAX_TRAIN_ROWS = 200_000  # filas de train para LR/NB/RF (era 30_000)
RF_SEARCH_ROWS = 60_000   # filas para la búsqueda de hiperparámetros del RF

NGRAM_WINDOW = 12        # últimos N caracteres sobre los que se sacan n-gramas
NGRAM_RANGE = (2, 4)     # n-gramas de 2 a 4 caracteres
LOOKUP_K = 3             # orden máximo de la tabla de búsqueda (backoff)
AUC_MAX_ROWS = 50_000    # tope de filas para el AUC-ROC (submuestreo fijo)
PROBA_CHUNK = 100_000     # filas por bloque al calcular probabilidades
SMOOTHING_ALPHA = 0.01   # suavizado aditivo para el log-loss del lookup


def timed(label: str):
  """Imprime y devuelve el reloj, para medir cada etapa."""
  print(f"[{datetime.now().strftime('%H:%M:%S')}] {label}")
  return time.time()


# -------------------------------------------------------------------------
# 2. Muestreo y features
# -------------------------------------------------------------------------
def context_matrix(series: pd.Series, ctx_length: int) -> np.ndarray:
  """Extrae los últimos `ctx_length` caracteres de cada texto en la serie."""
  characters = [list(str(text)[-ctx_length:]) for text in series]
  return np.asarray(characters, dtype=object)


def stratified_train_sample(
    train_df: pd.DataFrame, n_rows: int
) -> pd.DataFrame:
  """Muestra `n_rows` filas de train MANTENIENDO la distribución de clases.

  Antes se usaba df.sample(...) aleatorio: con una muestra pequeña, las
  clases raras del objetivo podían quedar fuera por completo. La
  estratificación garantiza que toda clase con suficientes filas en train
  está representada en la muestra (las clases con 1-2 filas pueden seguir
  fuera: es inevitable al submuestrear).
  """
  if n_rows >= len(train_df):
    return train_df
  # OJO: sklearn exige >= 2 filas por estrato para poder estratificar. En
  # train hay clases con UNA sola fila (los dígitos '1' y '7'; ver
  # train_char_counts_y en vocab.json), así que estratificar directo lanza
  # ValueError. Esas filas se agrupan en un estrato común (-1); y si el
  # estrato quedara con una sola fila, se mezcla con la clase más pequeña
  # que sí tenga >= 2 (distorsión de una fila: irrelevante).
  counts = train_df["y_id"].value_counts()
  too_few = counts[counts < 2].index
  strat = train_df["y_id"].mask(train_df["y_id"].isin(too_few), -1)
  if int((strat == -1).sum()) == 1:
    smallest = counts[counts >= 2].index[-1]
    strat = strat.mask(strat == -1, smallest)
  sample, _ = train_test_split(
      train_df,
      train_size=n_rows,
      stratify=strat,
      random_state=SEED,
  )
  return sample.reset_index(drop=True)


def build_features(
    train_sample: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ctx_length: int,
) -> dict:
  """Construye las dos representaciones de entrada.

  - "pos":      one-hot posicional de los últimos `ctx_length` caracteres
                (~76 x ctx_length columnas). Para RF: los árboles con cientos
                de miles de columnas de n-gramas serían lentísimos, así que
                el RF usa solo esta parte.
  - "combined": "pos" + conteos de n-gramas de caracteres (2-4) de los
                últimos NGRAM_WINDOW caracteres. Para LR y NB: aquí están
                los pares/tríos ('qu', 'Th', 'e t'...) que el one-hot
                posicional no puede representar.

  El OneHotEncoder y el CountVectorizer se ajustan SOLO con la muestra de
  train (fit en train, transform en val/test): sin fuga de información.
  """
  t0 = timed(f"Extrayendo features: one-hot posicional k={ctx_length} "
             f"+ n-gramas {NGRAM_RANGE} sobre los últimos {NGRAM_WINDOW} chars")

  encoder = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
  vectorizer = CountVectorizer(
      analyzer="char",
      ngram_range=NGRAM_RANGE,
      # lowercase=False es imprescindible aquí: el default (True) colapsaría
      # 'Th' con 'th', 'Qu' con 'qu', etc., y las mayúsculas son clases del
      # objetivo (79 clases con mayúsculas y minúsculas distintas).
      lowercase=False,
      min_df=3,
      max_features=200_000,
      dtype=np.float32,
  )

  pos_train = encoder.fit_transform(
      context_matrix(train_sample["x"], ctx_length)
  ).tocsr()
  pos_val = encoder.transform(context_matrix(val_df["x"], ctx_length)).tocsr()
  pos_test = encoder.transform(context_matrix(test_df["x"], ctx_length)).tocsr()

  ng_train = vectorizer.fit_transform(train_sample["x"].str[-NGRAM_WINDOW:])
  ng_val = vectorizer.transform(val_df["x"].str[-NGRAM_WINDOW:])
  ng_test = vectorizer.transform(test_df["x"].str[-NGRAM_WINDOW:])

  combined_train = sparse.hstack([pos_train, ng_train]).tocsr()
  combined_val = sparse.hstack([pos_val, ng_val]).tocsr()
  combined_test = sparse.hstack([pos_test, ng_test]).tocsr()

  print(
      f"  pos: {pos_train.shape[1]} columnas | "
      f"n-gramas: {ng_train.shape[1]} columnas | "
      f"train: {combined_train.shape}, val: {combined_val.shape}, "
      f"test: {combined_test.shape}"
  )
  print(f"  features listas en {time.time() - t0:.1f}s")

  return {
      "pos": (pos_train, pos_val, pos_test),
      "combined": (combined_train, combined_val, combined_test),
      "encoder": encoder,
  }


# -------------------------------------------------------------------------
# 3. Métricas
# -------------------------------------------------------------------------
def base_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
  """Métricas estándar de clasificación multiclase."""
  return {
      "Accuracy": accuracy_score(y_true, y_pred),
      "Precision (Macro)": precision_score(
          y_true, y_pred, average="macro", zero_division=0
      ),
      "Precision (Weighted)": precision_score(
          y_true, y_pred, average="weighted", zero_division=0
      ),
      "Recall (Macro)": recall_score(
          y_true, y_pred, average="macro", zero_division=0
      ),
      "Recall (Weighted)": recall_score(
          y_true, y_pred, average="weighted", zero_division=0
      ),
      "F1 (Macro)": f1_score(y_true, y_pred, average="macro", zero_division=0),
      "F1 (Weighted)": f1_score(
          y_true, y_pred, average="weighted", zero_division=0
      ),
      "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
  }


def finalize_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    p_true: np.ndarray | None = None,
    in_top3: np.ndarray | None = None,
) -> dict:
  """Métricas base + log-loss en bits y top-3 accuracy si hay probabilidades.

  El log-loss se reporta en BITS para comparar directamente con la entropía
  del objetivo medida en el EDA: H(y) = 4.5002 bits. Un modelo que no ha
  aprendido nada del contexto debería quedar cerca de ese número.
  """
  metrics = base_classification_metrics(y_true, y_pred)
  if p_true is not None:
    metrics["Log-Loss (bits)"] = float(
        -np.log(np.clip(p_true, 1e-15, 1.0)).mean() / np.log(2)
    )
  if in_top3 is not None:
    metrics["Top-3 Accuracy"] = float(in_top3.mean())
  return metrics


def class_columns(classes: np.ndarray, y_true: np.ndarray) -> np.ndarray:
  """Columna de predict_proba que corresponde a cada y_true (-1 si la clase
  no estaba en train: el modelo le asigna probabilidad 0, que es lo honesto)."""
  classes = np.asarray(classes)
  pos = np.searchsorted(classes, y_true)
  pos = np.minimum(pos, len(classes) - 1)
  valid = classes[pos] == y_true
  return np.where(valid, pos, -1)


def model_probabilities(
    model, x_data, y_true: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  """p_true (probabilidad asignada al carácter correcto) y acierto en top-3.

  Se calcula por bloques para no materializar la matriz completa de
  probabilidades (n_filas x 76 clases) en memoria.
  """
  col = class_columns(model.classes_, y_true)
  n = len(y_true)
  p_true = np.empty(n, dtype=np.float64)
  in_top3 = np.empty(n, dtype=bool)

  for start in range(0, n, PROBA_CHUNK):
    end = min(start + PROBA_CHUNK, n)
    proba = model.predict_proba(x_data[start:end])
    block_col = col[start:end]
    rows = np.arange(end - start)
    p_true[start:end] = np.where(
        block_col >= 0, proba[rows, np.clip(block_col, 0, None)], 0.0
    )
    top3 = np.argpartition(-proba, 2, axis=1)[:, :3]
    in_top3[start:end] = (top3 == block_col[:, None]).any(axis=1)

  return p_true, in_top3


def compute_macro_roc_auc_ovr(
    model, x_data, y_true: np.ndarray
) -> tuple[float, int, int]:
  """Calcula el Macro AUC-ROC (One-vs-Rest) adaptado a las clases presentes.

  Para que el costo no explote con val/test completos (76 AUCs binarios sobre
  cientos de miles de filas), se calcula sobre como máximo AUC_MAX_ROWS filas
  elegidas con semilla fija: es una métrica de ordenamiento y una muestra de
  50k la estima con la misma precisión práctica.
  """
  n = x_data.shape[0]
  if n > AUC_MAX_ROWS:
    rng = np.random.RandomState(SEED)
    keep = np.sort(rng.choice(n, AUC_MAX_ROWS, replace=False))
    x_data = x_data[keep]
    y_true = y_true[keep]

  y_proba = model.predict_proba(x_data)
  model_classes = np.asarray(model.classes_)

  known_mask = np.isin(y_true, model_classes)
  y_true_filtered = y_true[known_mask]
  y_proba_filtered = y_proba[known_mask]

  class_to_index = {int(cls): idx for idx, cls in enumerate(model_classes)}
  y_true_index = np.array([class_to_index[int(cls)] for cls in y_true_filtered])

  present_classes = np.unique(y_true_index)
  if len(present_classes) < 2:
    return np.nan, len(y_true_filtered), len(present_classes)

  y_true_binary = np.zeros((len(y_true_index), len(present_classes)))
  for i, cls_idx in enumerate(present_classes):
    y_true_binary[:, i] = y_true_index == cls_idx

  y_proba_present = y_proba_filtered[:, present_classes]

  # Re-normalizar probabilidades a las clases presentes si es necesario
  row_sums = y_proba_present.sum(axis=1, keepdims=True)
  row_sums[row_sums == 0] = 1.0
  y_proba_present = y_proba_present / row_sums

  macro_auc = roc_auc_score(
      y_true_binary, y_proba_present, average="macro", multi_class="ovr"
  )
  return macro_auc, len(y_true_filtered), len(present_classes)


def plot_and_save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    output_path: Path,
    writer=None,
    tag="confusion_matrix",
    labels=None,
):
  """Genera y guarda la matriz de confusión tanto en disco como en TensorBoard.

  `labels` fija el tamaño de la matriz (todas las clases del vocabulario), así
  las matrices de distintos modelos/splits son comparables posición a posición.
  """
  cm = confusion_matrix(y_true, y_pred, labels=labels)
  fig, ax = plt.subplots(figsize=(10, 8))
  cax = ax.imshow(cm, interpolation="nearest", cmap="viridis")
  ax.set_title(title)
  ax.set_xlabel("Predicted class")
  ax.set_ylabel("True class")
  fig.colorbar(cax)
  fig.tight_layout()

  fig.savefig(output_path, dpi=300)

  if writer is not None:
    writer.add_figure(tag, fig, global_step=0)

  plt.close(fig)
  print(f"Confusion matrix guardada en: {output_path}")


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    split_name: str,
    labels=None,
    target_names=None,
) -> None:
  """Guarda el reporte por clase (recall de clases raras incluido) a un .txt.

  El README pide reportar el recall por clase de las clases raras; por
  consola sería ilegible (76 clases), así que va a un archivo del run. Con
  `labels`/`target_names` (las 76 clases del vocabulario) el reporte lista
  TODAS las clases, incluidas las que no aparecen en el split: lo que pasa
  con las raras no puede quedar escondido.
  """
  report = classification_report(
      y_true, y_pred, labels=labels, target_names=target_names, zero_division=0
  )
  path = CURRENT_RUN_DIR / f"classification_report_{model_name}_{split_name}.txt"
  path.write_text(report, encoding="utf-8")


def log_metrics_to_writer(
    writer, model_name: str, split_name: str, metrics: dict
):
  """Registra un diccionario de métricas en TensorBoard."""
  if writer is None:
    return
  for metric_key, val in metrics.items():
    if not np.isnan(val):
      sanitized_key = metric_key.replace(" ", "_").replace("-", "_")
      writer.add_scalar(
          f"{model_name}/{split_name}_{sanitized_key}", val, global_step=0
      )


# -------------------------------------------------------------------------
# 4. Baselines de referencia (sin aprendizaje iterativo)
# -------------------------------------------------------------------------
def smoothed(counts: np.ndarray) -> np.ndarray:
  """Distribución con suavizado aditivo (no cambia el argmax)."""
  return (counts + SMOOTHING_ALPHA) / (
      counts.sum() + SMOOTHING_ALPHA * len(counts)
  )


def majority_baseline(
    y_true: np.ndarray, global_counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Siempre predice la clase más frecuente (' ').

  Es la referencia mínima: su accuracy es la frecuencia del espacio
  (~18.74% en el dataset completo). Un modelo que no la supere no aprendió
  nada. Sin suavizado el log-loss sería infinito (asigna 0 a las demás
  clases), por eso se usa la distribución suavizada.
  """
  dist = smoothed(global_counts)
  majority = int(np.argmax(global_counts))
  top3 = np.argsort(-global_counts)[:3]
  y_pred = np.full(len(y_true), majority, dtype=np.int64)
  p_true = dist[y_true]
  in_top3 = np.isin(y_true, top3)
  return y_pred, p_true, in_top3


def build_lookup_tables(
    train_df: pd.DataFrame, k_max: int, n_classes: int
) -> dict[int, dict[str, np.ndarray]]:
  """Tablas de conteo P(y | últimos k caracteres) sobre TODO el train.

  Con k=3 hay ~15 mil contextos distintos (EDA): es solo contar, así que se
  usa el train completo (2.99M filas) y no la muestra. Los modelos
  entrenados con 200k filas deben superar esto; si no, no están aprendiendo
  nada que una tabla de búsqueda no dé gratis.
  """
  tables: dict[int, dict[str, np.ndarray]] = {}

  for k in range(1, k_max + 1):
    t0 = timed(f"Construyendo tabla de búsqueda k={k} sobre TODO el train")
    ctx = train_df["x"].str[-k:]
    grouped = train_df.groupby(ctx)["y_id"].value_counts()

    table: dict[str, np.ndarray] = {}
    for (context, y_id), count in grouped.items():
      counts = table.get(context)
      if counts is None:
        counts = np.zeros(n_classes, dtype=np.float64)
        table[context] = counts
      counts[int(y_id)] = float(count)

    tables[k] = table
    total_rows = int(sum(c.sum() for c in table.values()))
    print(
        f"  k={k}: {len(table):,} contextos distintos, "
        f"{total_rows:,} filas cubiertas ({time.time() - t0:.1f}s)"
    )

  return tables


def lookup_table_predict(
    df: pd.DataFrame,
    tables: dict[int, dict[str, np.ndarray]],
    global_counts: np.ndarray,
    k_max: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Predicción con backoff: usa el contexto más largo visto en train.

  Para cada fila: prueba el contexto de k_max caracteres; si nunca apareció
  en train, retrocede a k_max-1, ... , a k=1, y si tampoco, usa la
  distribución global. La predicción es el argmax de la tabla que aplique.
  """
  y_true = df["y_id"].to_numpy()
  n = len(df)

  # Nivel base: distribución global (mayoría).
  dist_global = smoothed(global_counts)
  y_pred = np.full(n, int(np.argmax(global_counts)), dtype=np.int64)
  p_true = dist_global[y_true]
  top3_global = np.argsort(-global_counts)[:3]
  in_top3 = np.isin(y_true, top3_global)

  # Los niveles se aplican de superficial a profundo: el más largo gana.
  for k in range(1, k_max + 1):
    table = tables[k]
    ctx = df["x"].str[-k:]
    codes, uniques = pd.factorize(ctx)

    # Índices de filas por contexto único (una sola pasada por nivel).
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    boundaries = np.searchsorted(sorted_codes, np.arange(len(uniques)))
    boundaries = np.append(boundaries, n)

    for code in range(len(uniques)):
      context = str(uniques[code])
      counts = table.get(context)
      if counts is None:
        continue  # contexto no visto en train a este nivel: sigue el backoff

      rows = order[boundaries[code]:boundaries[code + 1]]
      dist = smoothed(counts)
      y_pred[rows] = int(np.argmax(counts))
      p_true[rows] = dist[y_true[rows]]
      top3 = np.argsort(-counts)[:3]
      in_top3[rows] = np.isin(y_true[rows], top3)

  return y_pred, p_true, in_top3


# -------------------------------------------------------------------------
# 5. Flujo Principal
# -------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
      prog="python -m src.baselines",
      description="Entrena y evalúa los modelos base sobre los splits fijos.",
  )
  parser.add_argument(
      "--train-rows", type=int, default=MAX_TRAIN_ROWS,
      help=f"filas de train para LR/NB/RF (default {MAX_TRAIN_ROWS})",
  )
  parser.add_argument(
      "--rf-search-rows", type=int, default=RF_SEARCH_ROWS,
      help=f"filas para el grid search del RF (default {RF_SEARCH_ROWS})",
  )
  parser.add_argument(
      "--context", type=int, default=CONTEXT_LENGTH,
      help=f"caracteres de contexto one-hot (default {CONTEXT_LENGTH})",
  )
  parser.add_argument(
      "--lookup-k", type=int, default=LOOKUP_K,
      help=f"orden máximo de la tabla de búsqueda (default {LOOKUP_K})",
  )
  args = parser.parse_args(argv)

  print("=" * 60)
  print(f"Inicializando ejecución. Logs guardados en: {CURRENT_RUN_DIR}")
  print(
      f"Parámetros: train-rows={args.train_rows:,}, "
      f"rf-search-rows={args.rf_search_rows:,}, context={args.context}, "
      f"lookup-k={args.lookup_k}"
  )
  print("=" * 60)

  writer = make_writer(CURRENT_RUN_DIR)

  # -------------------------------------------------------------------
  # 5.1 Cargar splits completos y vocabulario
  # -------------------------------------------------------------------
  t0 = timed("Cargando particiones del dataset (completas)...")
  train_df = load_split("train")
  val_df = load_split("val")
  test_df = load_split("test")
  vocab = read_vocab()
  n_classes = int(vocab["size"])
  # Para reportes/matrices completos: las 76 clases con su carácter real.
  all_labels = np.arange(n_classes)
  target_names = list(vocab["itos"])

  y_train_full = train_df["y_id"].to_numpy()
  global_counts = np.bincount(y_train_full, minlength=n_classes).astype(
      np.float64
  )

  print(
      f"train: {len(train_df):,} | val: {len(val_df):,} | "
      f"test: {len(test_df):,} | clases (vocab): {n_classes}"
  )

  # Muestreo ESTRATIFICADO de train para los modelos iterativos.
  train_sample = stratified_train_sample(train_df, args.train_rows)
  print(
      f"Muestra de train estratificada: {len(train_sample):,} filas "
      f"({len(train_sample) / len(train_df):.2%} del train completo)"
  )

  y_train = train_sample["y_id"].to_numpy()
  y_val = val_df["y_id"].to_numpy()
  y_test = test_df["y_id"].to_numpy()

  # -------------------------------------------------------------------
  # 5.2 Features
  # -------------------------------------------------------------------
  features = build_features(train_sample, val_df, test_df, args.context)
  x_train, x_val, x_test = features["combined"]
  pos_train, pos_val, pos_test = features["pos"]

  results: dict[str, dict[str, dict]] = {}

  def record(model_name: str, split_name: str, metrics: dict) -> None:
    results.setdefault(model_name, {})[split_name] = metrics
    log_metrics_to_writer(writer, model_name, split_name, metrics)

  # =====================================================================
  # BASELINE 0: CLASE MAYORITARIA (referencia mínima)
  # =====================================================================
  print("\n" + "=" * 60)
  print("Baseline 0: clase mayoritaria (predecir siempre ' ')...")
  print("=" * 60)

  for split_name, y_split in (("val", y_val), ("test", y_test)):
    y_pred, p_true, in_top3 = majority_baseline(y_split, global_counts)
    metrics = finalize_metrics(y_true=y_split, y_pred=y_pred,
                               p_true=p_true, in_top3=in_top3)
    record("Mayoría", split_name, metrics)
    print(
        f"  {split_name:5s}: accuracy {metrics['Accuracy']:.4f} | "
        f"log-loss {metrics['Log-Loss (bits)']:.4f} bits "
        f"(H(y) = 4.5002 bits)"
    )

  # =====================================================================
  # BASELINE 1: TABLA DE BÚSQUEDA N-GRAM CON BACKOFF
  # =====================================================================
  print("\n" + "=" * 60)
  print(f"Baseline 1: tabla de búsqueda (backoff k={args.lookup_k} -> 1)...")
  print("=" * 60)

  tables = build_lookup_tables(train_df, args.lookup_k, n_classes)

  for split_name, split_df, y_split in (
      ("val", val_df, y_val),
      ("test", test_df, y_test),
  ):
    y_pred, p_true, in_top3 = lookup_table_predict(
        split_df, tables, global_counts, args.lookup_k
    )
    metrics = finalize_metrics(y_true=y_split, y_pred=y_pred,
                               p_true=p_true, in_top3=in_top3)
    record(f"Lookup k={args.lookup_k}", split_name, metrics)
    print(
        f"  {split_name:5s}: accuracy {metrics['Accuracy']:.4f} | "
        f"top-3 {metrics['Top-3 Accuracy']:.4f} | "
        f"log-loss {metrics['Log-Loss (bits)']:.4f} bits"
    )
    save_classification_report(
        y_split, y_pred, f"lookup{args.lookup_k}", split_name,
        labels=all_labels, target_names=target_names,
    )
    plot_and_save_confusion_matrix(
        y_split, y_pred,
        f"Lookup k={args.lookup_k} - {split_name} Confusion Matrix",
        CURRENT_RUN_DIR / f"lookup_cm_{split_name}.png",
        writer=writer, tag=f"Lookup_{args.lookup_k}/CM_{split_name}",
        labels=all_labels,
    )

  # =====================================================================
  # BASELINE 2: REGRESIÓN LOGÍSTICA (features combinadas)
  # =====================================================================
  print("\n" + "=" * 60)
  print("Entrenando Baseline 2: Regresión Logística (one-hot + n-gramas)...")
  print("=" * 60)

  lr_search = []
  best_macro_f1_lr = -1.0
  best_lr_model = None
  best_lr_name = ""

  for C in (0.25, 1.0):
    t0 = timed(f"LR C={C}: ajustando sobre {x_train.shape[0]:,} filas")
    lr = LogisticRegression(
        C=C, solver="saga", max_iter=300, tol=1e-3, random_state=SEED,
    )
    lr.fit(x_train, y_train)
    y_val_pred_lr = lr.predict(x_val)
    macro_f1 = f1_score(y_val, y_val_pred_lr, average="macro", zero_division=0)
    lr_search.append(
        {"name": f"lr_C{C}", "val_macro_f1": macro_f1,
         "val_accuracy": accuracy_score(y_val, y_val_pred_lr)}
    )
    print(
        f"  LR C={C} -> Macro-F1 (val): {macro_f1:.4f} | "
        f"Acc (val): {lr_search[-1]['val_accuracy']:.4f} | "
        f"{time.time() - t0:.1f}s"
    )
    if macro_f1 > best_macro_f1_lr:
      best_macro_f1_lr, best_lr_model, best_lr_name = macro_f1, lr, f"lr_C{C}"

    if writer:
      writer.add_scalar("LR_GridSearch/Macro_F1", macro_f1, len(lr_search) - 1)

  pd.DataFrame(lr_search).to_csv(
      CURRENT_RUN_DIR / "lr_hyperparams_search.csv", index=False
  )
  print(f"Mejor LR: {best_lr_name} (Macro-F1 val: {best_macro_f1_lr:.4f})")

  for split_name, x_split, y_split in (
      ("val", x_val, y_val), ("test", x_test, y_test)
  ):
    y_pred_lr = best_lr_model.predict(x_split)
    p_true, in_top3 = model_probabilities(best_lr_model, x_split, y_split)
    metrics = finalize_metrics(y_split, y_pred_lr, p_true, in_top3)
    auc_lr, _, _ = compute_macro_roc_auc_ovr(best_lr_model, x_split, y_split)
    metrics["Macro AUC-ROC"] = auc_lr
    record("Regresión Logística", split_name, metrics)
    save_classification_report(
        y_split, y_pred_lr, "lr", split_name,
        labels=all_labels, target_names=target_names,
    )
    plot_and_save_confusion_matrix(
        y_split, y_pred_lr,
        f"Logistic Regression ({best_lr_name}) - {split_name} Confusion Matrix",
        CURRENT_RUN_DIR / f"lr_cm_{split_name}.png",
        writer=writer, tag=f"Logistic_Regression/CM_{split_name}",
        labels=all_labels,
    )

  print("\n--- Métricas Regresión Logística (Test) ---")
  for k, v in results["Regresión Logística"]["test"].items():
    print(f"{k:22s}: {v:.4f}")

  # =====================================================================
  # BASELINE 3: NAIVE BAYES (features combinadas)
  # =====================================================================
  print("\n" + "=" * 60)
  print("Entrenando Baseline 3: Naive Bayes (one-hot + n-gramas)...")
  print("=" * 60)

  nb_search = []
  best_macro_f1_nb = -1.0
  best_nb_model = None
  best_nb_name = ""

  for alpha in (0.05, 0.25, 1.0):
    t0 = timed(f"NB alpha={alpha}: ajustando")
    nb = MultinomialNB(alpha=alpha)
    nb.fit(x_train, y_train)
    y_val_pred_nb = nb.predict(x_val)
    macro_f1 = f1_score(y_val, y_val_pred_nb, average="macro", zero_division=0)
    nb_search.append(
        {"name": f"nb_alpha{alpha}", "val_macro_f1": macro_f1,
         "val_accuracy": accuracy_score(y_val, y_val_pred_nb)}
    )
    print(
        f"  NB alpha={alpha} -> Macro-F1 (val): {macro_f1:.4f} | "
        f"Acc (val): {nb_search[-1]['val_accuracy']:.4f} | "
        f"{time.time() - t0:.1f}s"
    )
    if macro_f1 > best_macro_f1_nb:
      best_macro_f1_nb, best_nb_model, best_nb_name = (
          macro_f1, nb, f"nb_alpha{alpha}"
      )
    if writer:
      writer.add_scalar("NB_GridSearch/Macro_F1", macro_f1, len(nb_search) - 1)

  pd.DataFrame(nb_search).to_csv(
      CURRENT_RUN_DIR / "nb_hyperparams_search.csv", index=False
  )
  print(f"Mejor NB: {best_nb_name} (Macro-F1 val: {best_macro_f1_nb:.4f})")

  for split_name, x_split, y_split in (
      ("val", x_val, y_val), ("test", x_test, y_test)
  ):
    y_pred_nb = best_nb_model.predict(x_split)
    p_true, in_top3 = model_probabilities(best_nb_model, x_split, y_split)
    metrics = finalize_metrics(y_split, y_pred_nb, p_true, in_top3)
    auc_nb, _, _ = compute_macro_roc_auc_ovr(best_nb_model, x_split, y_split)
    metrics["Macro AUC-ROC"] = auc_nb
    record("Naive Bayes", split_name, metrics)
    save_classification_report(
        y_split, y_pred_nb, "nb", split_name,
        labels=all_labels, target_names=target_names,
    )
    plot_and_save_confusion_matrix(
        y_split, y_pred_nb,
        f"Naive Bayes ({best_nb_name}) - {split_name} Confusion Matrix",
        CURRENT_RUN_DIR / f"nb_cm_{split_name}.png",
        writer=writer, tag=f"Naive_Bayes/CM_{split_name}",
        labels=all_labels,
    )

  print("\n--- Métricas clave Naive Bayes (Test) ---")
  for k, v in results["Naive Bayes"]["test"].items():
    print(f"{k:22s}: {v:.4f}")

  # Liberar memoria antes del RF: este modelo solo usa el one-hot posicional,
  # y las features combinadas ya no se necesitan (LR/NB terminaron). Con TODAS
  # las filas de train esto deja ~2 GB libres para los árboles del RF.
  features["combined"] = (None, None, None)
  del x_train, x_val, x_test
  gc.collect()

  # =====================================================================
  # BASELINE 4: RANDOM FOREST (búsqueda sobre subconjunto de train)
  # =====================================================================
  print("\n" + "=" * 60)
  print("Optimizando Baseline 4: Random Forest (solo features one-hot)...")
  print("=" * 60)
  print(
      "Nota: el RF usa solo el one-hot posicional. Con ~200k columnas de"
      " n-gramas cada árbol sería demasiado lento; además los árboles con"
      " features binarias necesitan mucha profundidad para reconstruir lo"
      " que un n-grama representa en una sola columna."
  )

  # El grid se ajusta sobre un SUBCONJUNTO estratificado de train; el ganador
  # se reajusta con todo el train disponible. Así el grid no cuesta una hora
  # y la selección no usa los datos con los que se reporta al final (test).
  # OJO: el subconjunto se transforma con el MISMO encoder ajustado en
  # train_sample (transform, no fit): sus filas no son las primeras N de
  # pos_train, así que no se puede recortar pos_train.
  rf_search_sample = stratified_train_sample(train_sample, args.rf_search_rows)
  pos_search = features["encoder"].transform(
      context_matrix(rf_search_sample["x"], args.context)
  ).tocsr()
  y_rf_search = rf_search_sample["y_id"].to_numpy()

  rf_grid = [
      {"n_estimators": 200, "max_depth": 16, "min_samples_leaf": 2,
       "max_features": "sqrt", "class_weight": None},
      {"n_estimators": 200, "max_depth": 24, "min_samples_leaf": 2,
       "max_features": "sqrt", "class_weight": None},
      {"n_estimators": 200, "max_depth": 16, "min_samples_leaf": 2,
       "max_features": "sqrt", "class_weight": "balanced_subsample"},
      {"n_estimators": 200, "max_depth": 24, "min_samples_leaf": 2,
       "max_features": "sqrt", "class_weight": "balanced_subsample"},
  ]

  rf_experiments = []
  best_macro_f1_rf = -1.0
  best_rf_config = None
  best_rf_name = ""

  for step, cfg in enumerate(rf_grid):
    cfg_name = "rf_" + "_".join(
        str(v) for v in cfg.values()
    ).replace("balanced_subsample", "bal")
    t0 = timed(f"{cfg_name}: ajustando sobre {len(rf_search_sample):,} filas")
    rf = RandomForestClassifier(n_jobs=-1, random_state=SEED, **cfg)
    rf.fit(pos_search, y_rf_search)
    val_pred_rf = rf.predict(pos_val)

    acc = accuracy_score(y_val, val_pred_rf)
    bal_acc = balanced_accuracy_score(y_val, val_pred_rf)
    macro_f1 = f1_score(y_val, val_pred_rf, average="macro", zero_division=0)
    weighted_f1 = f1_score(
        y_val, val_pred_rf, average="weighted", zero_division=0
    )

    if writer:
      writer.add_scalar("RF_GridSearch/Macro_F1", macro_f1, step)
      writer.add_scalar("RF_GridSearch/Accuracy", acc, step)

    rf_experiments.append({
        "name": cfg_name,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    })
    print(
        f"{cfg_name} -> Macro-F1 (val): {macro_f1:.4f} | Acc (val): "
        f"{acc:.4f} | {time.time() - t0:.1f}s"
    )

    if macro_f1 > best_macro_f1_rf:
      best_macro_f1_rf = macro_f1
      best_rf_config = cfg
      best_rf_name = cfg_name

  rf_results_df = (
      pd.DataFrame(rf_experiments)
      .sort_values("macro_f1", ascending=False)
      .reset_index(drop=True)
  )
  print("\nConfiguraciones de Random Forest (ordenadas por Macro-F1 en val):")
  print(rf_results_df.to_string(index=False))
  rf_results_df.to_csv(
      CURRENT_RUN_DIR / "rf_hyperparams_search.csv", index=False
  )

  # Reajuste del ganador con TODO el train disponible.
  print(
      f"\nReajustando el mejor RF ({best_rf_name}) con las "
      f"{len(train_sample):,} filas de train..."
  )
  t0 = timed(f"RF {best_rf_name}: reajuste final")
  best_rf_model = RandomForestClassifier(
      n_jobs=-1, random_state=SEED, **best_rf_config
  )
  best_rf_model.fit(pos_train, y_train)
  print(f"  reajuste listo en {time.time() - t0:.1f}s")

  for split_name, x_split, y_split in (
      ("val", pos_val, y_val), ("test", pos_test, y_test)
  ):
    y_pred_rf = best_rf_model.predict(x_split)
    p_true, in_top3 = model_probabilities(best_rf_model, x_split, y_split)
    metrics = finalize_metrics(y_split, y_pred_rf, p_true, in_top3)
    auc_rf, _, _ = compute_macro_roc_auc_ovr(best_rf_model, x_split, y_split)
    metrics["Macro AUC-ROC"] = auc_rf
    record("Random Forest", split_name, metrics)
    save_classification_report(
        y_split, y_pred_rf, "rf", split_name,
        labels=all_labels, target_names=target_names,
    )
    plot_and_save_confusion_matrix(
        y_split, y_pred_rf,
        f"Random Forest ({best_rf_name}) - {split_name} Confusion Matrix",
        CURRENT_RUN_DIR / f"rf_cm_{split_name}.png",
        writer=writer, tag=f"Best_Random_Forest/CM_{split_name}",
        labels=all_labels,
    )

  # =====================================================================
  # 6. Resumen Comparativo Final
  # =====================================================================
  print("\n" + "=" * 60)
  print("RESUMEN COMPARATIVO FINAL DE BASELINES")
  print("=" * 60)
  print(
      "Referencias: mayoría = 18.74% de accuracy; entropía H(y) = 4.5002"
      " bits (log-loss de un modelo sin información). Las métricas de val"
      " de LR/NB/RF son ligeramente optimistas (se seleccionaron sus"
      " hiperparámetros con val); las honestas son las de test."
  )

  metric_names = list(
      results["Regresión Logística"]["test"].keys()
  )
  columns = {"Métrica": metric_names}
  for model_name, splits in results.items():
    for split_name, metrics in splits.items():
      col_name = f"{model_name} ({split_name.capitalize()})"
      columns[col_name] = [metrics.get(m, np.nan) for m in metric_names]

  summary_df = pd.DataFrame(columns)
  print(summary_df.round(4).to_string(index=False))

  # Guardar en checkpoints/ y una copia en el run actual
  summary_df.to_csv(csv_output_path, index=False)
  summary_df.to_csv(CURRENT_RUN_DIR / "summary.csv", index=False)
  print(f"\nResultados guardados en '{csv_output_path}' y en '{CURRENT_RUN_DIR}'.")

  if writer:
    writer.close()
    print(f"\nLogs de TensorBoard finalizados en {CURRENT_RUN_DIR}")
    print("Para visualizarlos ejecuta: tensorboard --logdir=runs")

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
