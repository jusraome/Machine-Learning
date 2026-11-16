from typing import Literal
import pandas as pd

from src.config import PROCESSED_DIR, SPLIT_NAMES

SplitName = Literal["train", "val", "test"]

def load_split(name: SplitName) -> pd.DataFrame:
    """Carga un split de los datos ya procesados como un DataFrame de Pandas.
    
    Los archivos procesados (.parquet) ya contienen la limpieza, 
    las ventanas de 80 caracteres (x) y los índices de vocabulario listos (y_id).
    """
    if name not in SPLIT_NAMES:
        raise ValueError(f"Split desconocido {name!r}; se esperaba uno de {SPLIT_NAMES}")

    path = PROCESSED_DIR / f"{name}.parquet"
    
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}. "
            "Asegúrate de haber copiado la carpeta 'data/processed' del Taller 1."
        )
        
    return pd.read_parquet(path)