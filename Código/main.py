import os, yaml
from pathlib import Path
import numpy as np
import pandas as pd

from data_ptbxl import load_ptbxl, make_fixed_splits, build_loaders
from models import build_model
from trainer import train

def main():
    # Config
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Carpeta resultados por modelo
    BASE = Path(cfg["path"])
    model_name = (cfg.get("model", {}) or {}).get("name", "model")
    safe_name = "".join(c if (str(c).isalnum() or c in "-_.") else "_" for c in str(model_name)).lower()
    OUT = BASE / f"results_{safe_name}"
    OUT.mkdir(parents=True, exist_ok=True)
    print("Guardaré resultados en:", OUT.resolve())

    # Datos
    X, Y_df, mlb, class_names, label_col = load_ptbxl(cfg, OUT)
    print(f"Etiquetas usadas: {label_col} | #clases={len(class_names)}")
    print("Clases:", class_names)

    # Modelo / parámetros base 
    n_leads = 12
    sr = int(cfg["sampling_rate"])
    target_len = 1000 if sr == 100 else 5000
    model_tag = str(model_name)

    # Splits fijos por "strat_fold"
    splits = make_fixed_splits(Y_df)     # {"train": idx_train, "valid": idx_valid, "test": idx_test}
    idx_train = splits["train"]
    idx_valid = splits["valid"]

    fold_rows = []
    per_class = {"f1": [], "precision": [], "recall": [], "auroc": [], "ap": []}

    print("=" * 70)
    print("Entrenamiento con splits fijos: train=folds 1–8, valid=9")

    train_loader, valid_loader, n_classes = build_loaders(
        X, Y_df, mlb, idx_train, idx_valid, cfg, label_col=label_col
    )

    model = build_model(
        model_tag,
        n_classes=n_classes,
        n_leads=n_leads,
        sampling_rate=sr,
        target_len=target_len,
        dropout=float(cfg.get("dropout", 0.3)),
    )

    m = train(model, train_loader, valid_loader, "fixed", OUT, cfg)

    fold_rows.append(
        {
            "fold": "fixed",
            "f1_macro": m["f1_macro"],
            "precision_macro": m["precision_macro"],
            "recall_macro": m["recall_macro"],
            "auroc_macro": m["auroc_macro"],
            "ap_macro": m["ap_macro"],
            "hamming_loss": m["hamming_loss"],
            "subset_accuracy": m["subset_accuracy"],
        }
    )
    per_class["f1"].append(m["f1_per_class"])
    per_class["precision"].append(m["precision_per_class"])
    per_class["recall"].append(m["recall_per_class"])
    per_class["auroc"].append(m["auroc_per_class"])
    per_class["ap"].append(m["ap_per_class"])

    print(f"[Fixed] F1 macro (test) = {m['f1_macro']:.3f}")

    # Guardar CSV 
    df_folds = pd.DataFrame(fold_rows)
    df_folds.to_csv(OUT / "cv_fold_metrics.csv", index=False)

    f1_mean = np.nanmean(np.stack(per_class["f1"], axis=0), axis=0)
    precision_mean = np.nanmean(np.stack(per_class["precision"], axis=0), axis=0)
    recall_mean = np.nanmean(np.stack(per_class["recall"], axis=0), axis=0)
    auroc_mean = np.nanmean(np.stack(per_class["auroc"], axis=0), axis=0)
    ap_mean = np.nanmean(np.stack(per_class["ap"], axis=0), axis=0)

    # Prevalencia global por clase
    all_labels_bin = np.stack(list(Y_df["_labels_bin"].values))
    prevalence = all_labels_bin.mean(axis=0)

    df_per_class = pd.DataFrame(
        {
            "class": class_names,
            "prevalence": prevalence,
            "f1_mean": f1_mean,
            "precision_mean": precision_mean,
            "recall_mean": recall_mean,
            "auroc_mean": auroc_mean,
            "ap_mean": ap_mean,
        }
    )
    df_per_class.to_csv(OUT / "cv_per_class_mean.csv", index=False)

    print("\nCSV guardados en:")
    print(" -", (OUT / "cv_fold_metrics.csv").resolve())
    print(" -", (OUT / "cv_per_class_mean.csv").resolve())


if __name__ == "__main__":
    main()
