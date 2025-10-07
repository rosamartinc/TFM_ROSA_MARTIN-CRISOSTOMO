import os, ast, numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer


def _parse_scp(x):
    if pd.isna(x): return {}
    if isinstance(x, dict): return x
    x = str(x).strip()
    if x == "": return {}
    try: return ast.literal_eval(x)
    except Exception: return {}

def _aggregate_by(y_dic, mapping_df, col):
    s = set()
    for k in y_dic.keys():
        if k in mapping_df.index:
            val = mapping_df.loc[k, col]
            if isinstance(val, str) and val.strip():
                s.add(val)
    return list(s)

def _load_raw_data(df, sampling_rate, base_signals):
    files = df.filename_lr if sampling_rate == 100 else df.filename_hr
    base = Path(base_signals)
    X_list = []
    for f in files:
        stem = Path(str(f)).name            # Ejemplo: "records100/.../00001_lr" -> "00001_lr"
        p = base / f"{stem}_filtered.npy"
        sig = np.load(p, allow_pickle=False)  # esperado (L,12) o (12,L)
        if sig.ndim != 2:
            raise ValueError(f"Se esperaba 2D (L,12) o (12,L); llegó {sig.shape}")
        L, C = sig.shape
        if C == 12:
            pass  # (L,12) OK
        elif L == 12:
            sig = sig.T  # (12,L) -> (L,12)
        else:
            raise ValueError(f"La señal no tiene 12 canales: {sig.shape}")
        X_list.append(sig.astype(np.float32))
    X = np.stack(X_list, axis=0)  # (N, L, 12)
    X = np.transpose(X, (0, 2, 1))       # (N, 12, L)
    print(f"Cargadas {X.shape[0]} señales de ECG con forma {X.shape[1:]} cada una")
    return X

def load_ptbxl(cfg, out_dir: Path):
    base = cfg["path"]
    base_signals = cfg["path_signals"]
    sampling_rate = int(cfg["sampling_rate"])
    target_len = 1000 if sampling_rate == 100 else 5000

    md_path = os.path.join(base, cfg["metadata_file"])
    if md_path.endswith(".csv"):
        Y = pd.read_csv(md_path, sep=";")
    else:
        Y = pd.read_excel(md_path, header=0)
    Y.columns = Y.columns.str.strip()
    
    print("COLUMNAS METADATA:", list(Y.columns)[:20])
    
    Y["scp_codes"] = Y["scp_codes"].apply(_parse_scp)

    agg_df = pd.read_csv(os.path.join(base, "scp_statements.csv"), index_col=0)
    agg_df = agg_df[agg_df["diagnostic"] == 1]

    Y["diagnostic_superclass"] = Y["scp_codes"].apply(lambda d: _aggregate_by(d, agg_df, "diagnostic_class"))

    mlb = MultiLabelBinarizer()
    mlb.fit(list(Y["diagnostic_superclass"]))
    class_names = list(mlb.classes_)
    
    X = _load_raw_data(Y, sampling_rate, base_signals)

    Y['_labels_bin'] = list(mlb.transform(list(Y["diagnostic_superclass"])).astype(np.float32))
    return X, Y, mlb, class_names, "diagnostic_superclass"

def make_fixed_splits(Y_df, model_name="model"):

    import numpy as np

    # Entrenamiento: folds 1-8
    idx_train = np.where(Y_df.strat_fold.isin(range(1, 9)))[0]
    # Validación: fold 9
    idx_valid = np.where(Y_df.strat_fold == 9)[0]

    folds = {
        "train": idx_train,
        "valid": idx_valid,
    }

    return folds

def build_loaders(X, Y_df, mlb, idx_train, idx_valid, cfg, label_col="diagnostic_superclass"):
    import torch
    from torch.utils.data import Dataset, DataLoader

    class ECGDataset(torch.utils.data.Dataset):
        def __init__(self, X, Y_bin):
            self.X = X.astype(np.float32)
            self.Y = Y_bin.astype(np.float32)

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            x = self.X[idx]  # (L, 12)
            y = self.Y[idx]  # (C,)

            # Normalización z-score por derivación
            mean = np.nanmean(x, axis=0, keepdims=True)   # media por canal
            std  = np.nanstd(x,  axis=0, keepdims=True)   # desviación por canal
            std[std < 1e-6] = 1e-6
            x = (x - mean) / std
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            # Si tu modelo espera (12, L), descomenta:
            # x = x.T

            return torch.from_numpy(x), torch.from_numpy(y)

    # Etiquetas
    y_train = list(Y_df.iloc[idx_train][label_col])
    y_valid = list(Y_df.iloc[idx_valid][label_col])

    y_train_bin = mlb.transform(y_train).astype(np.float32)
    y_valid_bin = mlb.transform(y_valid).astype(np.float32)
    
    # Datasets
    train_ds = ECGDataset(X[idx_train], y_train_bin)
    valid_ds = ECGDataset(X[idx_valid], y_valid_bin)

    # Loaders
    bt = int(cfg.get('batch_train', 64))
    bv = int(cfg.get('batch_test', 128))

    train_loader = DataLoader(train_ds, batch_size=bt, shuffle=True,  num_workers=0, pin_memory=False)
    valid_loader = DataLoader(valid_ds, batch_size=bv, shuffle=False, num_workers=0, pin_memory=False)

    n_classes = y_train_bin.shape[1]
    return train_loader, valid_loader, n_classes
