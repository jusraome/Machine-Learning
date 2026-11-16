#!/usr/bin/env bash
# Flujo completo de entrenamiento (Fase 4):
#   1) TensorBoard en :6006 (solo si no está ya corriendo)
#   2) Pestaña de Firefox con las curvas de pérdida en vivo
#   3) Entrenamiento en background (nohup) con los args que se le pasen
#
# Uso:  ./scripts/train.sh --epochs 2
#       ./scripts/train.sh --epochs 1 --max_train_batches 20 --max_val_batches 5   # smoke test
set -euo pipefail
cd "$(dirname "$0")/.."

PORT=6006
TB_LOG=tensorboard.log
TRAIN_LOG=train.log

# 0) No permitir dos entrenamientos a la vez
if pgrep -f "src/train.py" >/dev/null 2>&1; then
  echo "Ya hay un entrenamiento corriendo (PID: $(pgrep -f 'src/train.py' | tr '\n' ' '))."
  echo "Detenlo primero con: pkill -f 'src/train.py'"
  exit 1
fi

# 1) TensorBoard solo si el puerto no responde
if ! curl -s "http://localhost:${PORT}" >/dev/null 2>&1; then
  nohup .venv/bin/tensorboard --logdir runs --port "${PORT}" > "${TB_LOG}" 2>&1 &
  echo "TensorBoard PID: $! -> http://localhost:${PORT}"
  sleep 5
else
  echo "TensorBoard ya está corriendo en http://localhost:${PORT}"
fi

# 2) Abrir pestaña de Firefox (solo si hay sesión gráfica)
if [[ -n "${DISPLAY:-}" ]] && command -v firefox >/dev/null 2>&1; then
  (firefox --new-tab "http://localhost:${PORT}" >/dev/null 2>&1 &)
  echo "Pestaña de Firefox abierta."
else
  echo "Sin sesión gráfica: abre manualmente http://localhost:${PORT}"
fi

# 3) Entrenamiento en background
nohup .venv/bin/python -u src/train.py "$@" > "${TRAIN_LOG}" 2>&1 &
echo "Entrenamiento PID: $! (seguir con: tail -f ${TRAIN_LOG})"
