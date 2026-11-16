import sys
from pathlib import Path
# Hyperparaametros del modelo

VOCAB_SIZE = 79         # Define cuantos simbolos unicos puede leer y predecir el modelo al igual que el tamaño del los embiddings y la capa lineal de salida
WINDOW_SIZE = 80        # Es el tamaño de cada entrada del dataset (le dice a la capa de positional encoding hasta que caracter debe contar)
D_MODEL = 256           # Define en cuántos números (dimensiones) se va a convertir cada carácter una vez tokenizado. Un d_model más grande permite al modelo aprender significados y reglas gramaticales más ricas, pero consume más memoria
NHEAD = 8               # Le dice al modelo cuántos "focos de atención" diferentes puede usar al mismo tiempo (a cuantas partes del texto le debe prestar atencion) (d_model % nhead = 0)
NUM_LAYERS = 6          # Cantidad de capas de la red neuronal, mas capas equivalen a entender mas patrones
DIM_FFW = 1024          # Cuantas neuronas debe tener cada capa oculta. Suele ser d_model*4
DROPOUT = 0.1           # Porcentaje de neuronas que se apagan en el entrenamiento para que el modelo no dependa de una sola neurona

# Hiperparametros entrenamiento

BATCH_SIZE = 128        # Cuantos ejemplos (entradas de 80 caracteres) pocesa la gpu/cpu al mismo tiempo antes de actualizar los pesos 
LEARNING_RATE = 1e-3    # Que tangrandes son los saltos del optimizador al ajustar los pesos de las neuronas calculando el error
NUM_EPOCHS = 100        # Cuántas veces el modelo va a ver el dataset de entrenamiento completo

# Ubicaciones de los datos y carpetas
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# Nombres de las Particiones
SPLIT_NAMES = ("train", "val", "test")