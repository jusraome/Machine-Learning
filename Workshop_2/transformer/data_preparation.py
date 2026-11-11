
import sys
import numpy as np 
from pathlib import Path


#Splitted data path


PROJECT_ROOT = Path.cwd()
OUTPUT_DIR = PROJECT_ROOT / "Workshop_2" / "data"

if not (PROJECT_ROOT / "Workshop_1" / "src" / "dataset.py").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from Workshop_1.src.dataset import load_split

#tokenizer path
if not (PROJECT_ROOT / "Workshop_2" / "transformer").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

TOKENIZER_DIR = PROJECT_ROOT / "Workshop_2" / "transformer"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Workshop_2.transformer.tokenizer import Tokenizer

train_df = load_split("train")
test_df = load_split("test")
val_df = load_split("val")

token = Tokenizer()

#print(len(train_df), list(train_df.columns)) #load verification 

#!train split x axis
X_train = train_df['x']

Xt_encoded = X_train.apply(token.encode)
print(f"X train encoded preview \n{Xt_encoded.head(5)}")

#!test split x axix 

X_test = test_df['x']

Xte_encoded = X_test.apply(token.encode)
print(f"X test encoded preview \n{Xte_encoded.head(5)}")


#!val split x axis 

X_val = val_df['x']

Xv_encoded = X_val.apply(token.encode)
print(f"X val encoded preview {Xv_encoded.head(5)}")


# save  tokenized data 

OUTPUT_DIR.mkdir(parents= True, exist_ok=True)

def save_split(df, x_encoded, split_name): 
    x = np.asarray(x_encoded.tolist(), dtype= np.int16)
    y = df["y_id"].to_numpy(dtype=np.int16)
    np.savez_compressed(
        OUTPUT_DIR / f"{split_name}.npz", 
        X= x,
        y = y
    )

save_split(train_df,Xt_encoded, "train")
save_split(test_df,Xte_encoded, "test")
save_split(val_df,Xv_encoded, "val")




