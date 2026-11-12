"""baselines.py

Consolidación y evaluación de modelos base (Regresión Logística, Random Forest y
Naive Bayes)
para predicción de caracteres, con soporte de logging en 'runs/'.
"""

from datetime import datetime
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OneHotEncoder

# Intento de importación de SummaryWriter (PyTorch o TensorBoard puro)
try:
  from torch.utils.tensorboard import SummaryWriter
except ImportError:
  try:
    from tensorboard.summary import Writer as SummaryWriter
  except ImportError:
    SummaryWriter = None
    print(
        "Advertencia: No se encontró 'tensorboard' ni 'torch'. Ejecuta: pip"
        " install tensorboard"
    )

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
  from src.dataset import load_split

  set_seeds()
except ImportError:
  try:
    from config import SEED, set_seeds
    from dataset import load_split

    set_seeds()
  except ImportError as e:
    raise ImportError(
        f"No se pudo encontrar 'dataset.py' ni 'config.py'. "
        f"Verifica que existan en '{SRC_DIR}' o '{PROJECT_ROOT}'. Detalle: {e}"
    )

# Parámetros globales
CONTEXT_LENGTH = 5
MAX_TRAIN_ROWS = 30000
MAX_VAL_ROWS = 10000
MAX_TEST_ROWS = 10000


# -------------------------------------------------------------------------
# 2. Funciones Auxiliares y de Logging
# -------------------------------------------------------------------------
def context_matrix(series: pd.Series, ctx_length: int) -> np.ndarray:
  """Extrae los últimos `ctx_length` caracteres de cada texto en la serie."""
  characters = [list(str(text)[-ctx_length:]) for text in series]
  return np.asarray(characters, dtype=object)


def evaluate_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict:
  """Calcula métricas estándar de clasificación multiclase."""
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


def compute_macro_roc_auc_ovr(
    model, x_data, y_true: np.ndarray
) -> tuple[float, int, int]:
  """Calcula el Macro AUC-ROC (One-vs-Rest) adaptado a clases presentes

  en la partición de prueba/validación.
  """
  y_proba = model.predict_proba(x_data)
  model_classes = model.classes_

  known_mask = np.isin(y_true, model_classes)
  y_true_filtered = y_true[known_mask]
  y_proba_filtered = y_proba[known_mask]

  class_to_index = {class_id: idx for idx, class_id in enumerate(model_classes)}
  y_true_index = np.array([class_to_index[cls] for cls in y_true_filtered])

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
):
  """Genera y guarda la matriz de confusión tanto en disco como en TensorBoard."""
  cm = confusion_matrix(y_true, y_pred)
  fig, ax = plt.subplots(figsize=(10, 8))
  cax = ax.imshow(cm, interpolation="nearest", cmap="viridis")
  ax.set_title(title)
  ax.set_xlabel("Predicted class")
  ax.set_ylabel("True class")
  fig.colorbar(cax)
  fig.tight_layout()

  # Guardar copia física en la carpeta del run
  fig.savefig(output_path, dpi=300)

  # Registrar en TensorBoard si está habilitado
  if writer is not None:
    writer.add_figure(tag, fig, global_step=0)

  plt.close(fig)
  print(f"Confusion matrix guardada en: {output_path}")


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
# 3. Flujo Principal
# -------------------------------------------------------------------------
def main():
  print("=" * 60)
  print(f"Inicializando ejecucion. Logs guardados en: {CURRENT_RUN_DIR}")
  print("=" * 60)

  writer = (
      SummaryWriter(log_dir=str(CURRENT_RUN_DIR)) if SummaryWriter else None
  )

  print("Cargando particiones del dataset...")
  train_df = load_split("train")
  val_df = load_split("val")
  test_df = load_split("test")

  # Muestreo controlado por SEED
  train_sample = train_df.sample(
      n=min(MAX_TRAIN_ROWS, len(train_df)), random_state=SEED
  )
  val_sample = val_df.sample(
      n=min(MAX_VAL_ROWS, len(val_df)), random_state=SEED
  )
  test_sample = test_df.sample(
      n=min(MAX_TEST_ROWS, len(test_df)), random_state=SEED
  )

  print(f"Train sample: {len(train_sample)} filas")
  print(f"Val sample:   {len(val_sample)} filas")
  print(f"Test sample:  {len(test_sample)} filas")

  # ---------------------------------------------------------------------
  # 4. Extracción de Features (Context Window) y One-Hot Encoding
  # ---------------------------------------------------------------------
  print("\nProcesando contexto de caracteres...")
  x_train_chars = context_matrix(train_sample["x"], CONTEXT_LENGTH)
  x_val_chars = context_matrix(val_sample["x"], CONTEXT_LENGTH)
  x_test_chars = context_matrix(test_sample["x"], CONTEXT_LENGTH)

  encoder = OneHotEncoder(handle_unknown="ignore")
  x_train = encoder.fit_transform(x_train_chars)
  x_val = encoder.transform(x_val_chars)
  x_test = encoder.transform(x_test_chars)

  y_train = train_sample["y_id"].to_numpy()
  y_val = val_sample["y_id"].to_numpy()
  y_test = test_sample["y_id"].to_numpy()

  print(f"Dimensiones X_train codificado: {x_train.shape}")
  print(f"Dimensiones X_val codificado:   {x_val.shape}")
  print(f"Dimensiones X_test codificado:  {x_test.shape}")

  # =====================================================================
  # BASELINE 1: REGRESIÓN LOGÍSTICA
  # =====================================================================
  print("\n" + "=" * 60)
  print("Entrenando Baseline 1: Regresion Logistica...")
  print("=" * 60)

  lr_model = LogisticRegression(
      C=1.0, solver="saga", max_iter=1000, random_state=SEED, n_jobs=-1
  )
  lr_model.fit(x_train, y_train)

  # Validación
  y_val_pred_lr = lr_model.predict(x_val)
  val_metrics_lr = evaluate_classification_metrics(y_val, y_val_pred_lr)
  val_auc_lr, _, _ = compute_macro_roc_auc_ovr(lr_model, x_val, y_val)
  val_metrics_lr["Macro AUC-ROC"] = val_auc_lr
  log_metrics_to_writer(writer, "Logistic_Regression", "val", val_metrics_lr)

  # Test
  y_test_pred_lr = lr_model.predict(x_test)
  test_metrics_lr = evaluate_classification_metrics(y_test, y_test_pred_lr)
  test_auc_lr, _, _ = compute_macro_roc_auc_ovr(lr_model, x_test, y_test)
  test_metrics_lr["Macro AUC-ROC"] = test_auc_lr
  log_metrics_to_writer(writer, "Logistic_Regression", "test", test_metrics_lr)

  print("\n--- Metricas Regresion Logistica (Test) ---")
  for k, v in test_metrics_lr.items():
    print(f"{k:22s}: {v:.4f}")

  plot_and_save_confusion_matrix(
      y_val,
      y_val_pred_lr,
      "Logistic Regression - Validation Confusion Matrix",
      CURRENT_RUN_DIR / "lr_cm_validation.png",
      writer=writer,
      tag="Logistic_Regression/CM_Validation",
  )
  plot_and_save_confusion_matrix(
      y_test,
      y_test_pred_lr,
      "Logistic Regression - Test Confusion Matrix",
      CURRENT_RUN_DIR / "lr_cm_test.png",
      writer=writer,
      tag="Logistic_Regression/CM_Test",
  )

  # =====================================================================
  # BASELINE 2: RANDOM FOREST (BÚSQUEDA DE HIPERPARÁMETROS)
  # =====================================================================
  print("\n" + "=" * 60)
  print("Optimizando Baseline 2: Random Forest...")
  print("=" * 60)

  n_estimators_list = [100, 200, 400]
  depth_list = [12, 20, 24]
  min_leaf_list = [2, 5]
  max_feat_list = ["sqrt", 0.5]
  class_weights_list = ["balanced", "balanced_subsample"]

  rf_experiments = []
  best_macro_f1 = -1.0
  best_rf_model = None
  best_rf_name = ""

  step = 0
  for n in n_estimators_list:
    for d in depth_list:
      for l in min_leaf_list:
        for f in max_feat_list:
          for w in class_weights_list:
            cfg_name = f"rf_{n}_{d}_{l}_{f}_{w}"
            rf = RandomForestClassifier(
                n_estimators=n,
                max_depth=d,
                min_samples_leaf=l,
                max_features=f,
                class_weight=w,
                n_jobs=-1,
                random_state=SEED,
            )
            rf.fit(x_train, y_train)
            val_pred_rf = rf.predict(x_val)

            acc = accuracy_score(y_val, val_pred_rf)
            bal_acc = balanced_accuracy_score(y_val, val_pred_rf)
            macro_f1 = f1_score(
                y_val, val_pred_rf, average="macro", zero_division=0
            )
            weighted_f1 = f1_score(
                y_val, val_pred_rf, average="weighted", zero_division=0
            )

            # Log opcional del grid search en TensorBoard
            if writer:
              writer.add_scalar("RF_GridSearch/Macro_F1", macro_f1, step)
              writer.add_scalar("RF_GridSearch/Accuracy", acc, step)
            step += 1

            rf_experiments.append({
                "name": cfg_name,
                "accuracy": acc,
                "balanced_accuracy": bal_acc,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
            })

            if macro_f1 > best_macro_f1:
              best_macro_f1 = macro_f1
              best_rf_model = rf
              best_rf_name = cfg_name

            print(f"{cfg_name} -> Macro-F1: {macro_f1:.4f} | Acc: {acc:.4f}")

  rf_results_df = (
      pd.DataFrame(rf_experiments)
      .sort_values("macro_f1", ascending=False)
      .reset_index(drop=True)
  )
  print("\nTop 5 Configuraciones de Random Forest:")
  print(rf_results_df.head(5).to_string())

  # Guardar resultados de la búsqueda de hiperparámetros en runs/
  rf_results_df.to_csv(CURRENT_RUN_DIR / "rf_hyperparams_search.csv", index=False)

  # Evaluación del mejor Random Forest
  print(
      f"\nEvaluando el mejor modelo RF ({best_rf_name}) en Validation y Test..."
  )
  y_val_pred_rf = best_rf_model.predict(x_val)
  val_metrics_rf = evaluate_classification_metrics(y_val, y_val_pred_rf)
  val_auc_rf, _, _ = compute_macro_roc_auc_ovr(best_rf_model, x_val, y_val)
  val_metrics_rf["Macro AUC-ROC"] = val_auc_rf
  log_metrics_to_writer(writer, "Best_Random_Forest", "val", val_metrics_rf)

  y_test_pred_rf = best_rf_model.predict(x_test)
  test_metrics_rf = evaluate_classification_metrics(y_test, y_test_pred_rf)
  test_auc_rf, _, _ = compute_macro_roc_auc_ovr(best_rf_model, x_test, y_test)
  test_metrics_rf["Macro AUC-ROC"] = test_auc_rf
  log_metrics_to_writer(writer, "Best_Random_Forest", "test", test_metrics_rf)


  plot_and_save_confusion_matrix(
        y_val,
        y_val_pred_rf,
        f"Random Forest ({best_rf_name}) - Validation Confusion Matrix",
        CURRENT_RUN_DIR / "rf_cm_validation.png",
        writer=writer,
        tag="Best_Random_Forest/CM_validation",
    )
  
  plot_and_save_confusion_matrix(
      y_test,
      y_test_pred_rf,
      f"Random Forest ({best_rf_name}) - Test Confusion Matrix",
      CURRENT_RUN_DIR / "rf_cm_test.png",
      writer=writer,
      tag="Best_Random_Forest/CM_Test",
  )

  # =====================================================================
  # BASELINE 3: NAIVE BAYES
  # =====================================================================
  print("\n" + "=" * 60)
  print("Entrenando Baseline 3: Naive Bayes...")
  print("=" * 60)

  nb_model = MultinomialNB(alpha=0.1)
  nb_model.fit(x_train, y_train)

  y_val_pred_nb = nb_model.predict(x_val)
  val_metrics_nb = evaluate_classification_metrics(y_val, y_val_pred_nb)
  val_auc_nb, _, _ = compute_macro_roc_auc_ovr(nb_model, x_val, y_val)
  val_metrics_nb["Macro AUC-ROC"] = val_auc_nb
  log_metrics_to_writer(writer, "Naive_Bayes", "val", val_metrics_nb)

  y_test_pred_nb = nb_model.predict(x_test)
  test_metrics_nb = evaluate_classification_metrics(y_test, y_test_pred_nb)
  test_auc_nb, _, _ = compute_macro_roc_auc_ovr(nb_model, x_test, y_test)
  test_metrics_nb["Macro AUC-ROC"] = test_auc_nb
  log_metrics_to_writer(writer, "Naive_Bayes", "test", test_metrics_nb)

  print("\n--- Metricas clave Naive Bayes (Test) ---")
  for k, v in test_metrics_nb.items():
    print(f"{k:22s}: {v:.4f}")

  plot_and_save_confusion_matrix(
        y_val,
        y_val_pred_nb,
        "Naive Bayes - Validation Confusion Matrix",
        CURRENT_RUN_DIR / "nb_cm_validation.png",
        writer=writer,
        tag="Naive_Bayes/CM_validation",
    )

  plot_and_save_confusion_matrix(
      y_test,
      y_test_pred_nb,
      "Naive Bayes - Test Confusion Matrix",
      CURRENT_RUN_DIR / "nb_cm_test.png",
      writer=writer,
      tag="Naive_Bayes/CM_Test",
  )

  # =====================================================================
  # 5. Resumen Comparativo Final
  # =====================================================================
  print("\n" + "=" * 60)
  print("RESUMEN COMPARATIVO FINAL DE BASELINES")
  print("=" * 60)

  summary_df = pd.DataFrame({
      "Métrica": list(test_metrics_lr.keys()),
      "LR (Val)": [val_metrics_lr[m] for m in test_metrics_lr.keys()],
      "LR (Test)": [test_metrics_lr[m] for m in test_metrics_lr.keys()],
      "NB (Val)": [val_metrics_nb[m] for m in test_metrics_lr.keys()],
      "NB (Test)": [test_metrics_nb[m] for m in test_metrics_lr.keys()],
      "Best RF (Val)": [val_metrics_rf[m] for m in test_metrics_lr.keys()],
      "Best RF (Test)": [test_metrics_rf[m] for m in test_metrics_lr.keys()],
  })
  print(summary_df.round(4).to_string(index=False))

  # Guardar en checkpoints/ (como tenías) y una copia en el run actual
  summary_df.to_csv(csv_output_path, index=False)
  summary_df.to_csv(CURRENT_RUN_DIR / "summary.csv", index=False)
  print(f"\nResultados guardados en '{csv_output_path}' y en '{CURRENT_RUN_DIR}'.")

  # Cerrar TensorBoard Writer
  if writer:
    writer.close()
    print(f"\nLogs de TensorBoard finalizados en: {CURRENT_RUN_DIR}")
    print("Para visualizarlos ejecuta: tensorboard --logdir=runs")


if __name__ == "__main__":
  main()