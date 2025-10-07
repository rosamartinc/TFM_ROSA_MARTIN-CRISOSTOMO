import argparse
import pandas as pd
import torch
import numpy as np
import wfdb
from models import build_model 
from pathlib import Path


# Importamos filtrado del script clean_ecg_scipy
from clean_ecg_scipy import butter_bandpass_filter, notch_filter, remove_baseline_wander

DEFAULT_CLASS_NAMES = ["CD", "HYP", "MI", "NORM", "STTC"]

def clean_ecg_infer(signal: np.ndarray, fs: float,
                    lowcut: float = 0.5, highcut: float = 40.0, order: int = 4,
                    powerline: float = 50.0, q_notch: float = 30.0,
                    baseline: bool = True) -> np.ndarray:
   
    y = signal.astype(float)
    if baseline:
        y = remove_baseline_wander(y, fs, enable=True)
    y = butter_bandpass_filter(y, fs, lowcut=lowcut, highcut=highcut, order=order)
    y = notch_filter(y, fs, f0=powerline, q=q_notch)
    return y


def load_signal_wfdb(record_base: Path):
    
    # Carga un ECG en formato WFDB usando solo el 'basename' (sin extensión).
    #Devuelve (signal[N,C], fs)
    
    # wfdb.rdsamp necesita el path sin extensión
    base = str(record_base).replace(".dat", "").replace(".hea", "")
    sig, fields = wfdb.rdsamp(base)
    fs = float(fields.get("fs"))
    return sig, fs


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #SI SE DETECTA GPU SE USA

    print("Comenzando el proceso de inferencia para obtener superclase")
    print(f"- Excel: {args.excel_path}")
    print(f"- Base dir: {args.base_dir}")
    print(f"- Señal (filename_lr): {args.signal_name}")
    print(f"- Checkpoint: {args.checkpoint}")

    # Cargar Excel y localizar una señal
    df = pd.read_excel(args.excel_path)
    if "filename_lr" not in df.columns:
        raise ValueError("La columna 'filename_lr' no existe en el Excel.")

    row = df.loc[df["filename_lr"] == args.signal_name]
    if row.empty:
        raise ValueError(f"No se encontró '{args.signal_name}' en {args.excel_path}")

    # Ruta WFDB (sin extensión)
    wfdb_base = Path(args.base_dir) / row.iloc[0]["filename_lr"]

    # Cargar señal WFDB y filtrar 
    signal, fs = load_signal_wfdb(wfdb_base)
    print(fs)
    signal = clean_ecg_infer(signal, fs=fs,
                             lowcut=args.lowcut, highcut=args.highcut, order=args.order,
                             powerline=args.powerline, q_notch=args.q_notch,
                             baseline=not args.no_baseline)

    # Ajuste de forma para el modelo
    # (B, L, C).

    sig = signal
    if sig.ndim != 2:
        raise ValueError(f"Se esperaba 2D (L,12) o (12,L); llegó {sig.shape}")

    L, C = sig.shape
    if C == 12:
    # (L,12) OK
        sig = np.ascontiguousarray(sig, dtype=np.float32)     
    elif L == 12:
        # (12,L) -> (L,12)
        sig = np.ascontiguousarray(sig.T, dtype=np.float32)
    else:
        raise ValueError(f"La señal no tiene 12 canales: {sig.shape}")

    print ("(",L,",",C,")")
    
    # Batch y channels-first
    # (L,12) -> (1, L, 12) -> (1, 12, L)
    tensor = torch.from_numpy(sig).unsqueeze(0)          # (1, L, 12)
    tensor = tensor.permute(0, 2, 1).contiguous().to(device)  # (1, 12, L)

    print (tensor.shape)


    # Cargar modelo
    
    model = build_model(
        args.model_name,
        n_classes=5,
        n_leads=12,
        sampling_rate=fs,
        target_len=1000,
        dropout=float(0.3) #IGUAL QUE EN EL COFIG.YAML
    )
    
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])
    model = model.to(device).eval()

    # Inferencia
    class_names = DEFAULT_CLASS_NAMES

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1) if logits.shape[1] > 1 else torch.sigmoid(logits)

    probs_np = probs.squeeze(0).cpu().numpy()
    pred_idx = int(probs_np.argmax())
    pred_name = class_names[pred_idx]

    # Probabilidades ordenadas descendentemente (nombre, prob)
    rank = sorted(zip(class_names, probs_np.tolist()), key=lambda x: x[1], reverse=True)

    print("\n=== RESULTADO INFERENCIA SUPERCLASS ===")
    print(f"Señal (filename_lr): {args.signal_name}")
    print(f"Fs: {fs} Hz")
    print(f"Clase predicha (idx): {pred_idx}")
    print(f"Clase predicha (name): {pred_name}")
    print("Probabilidades por clase:")
    
    for name, p in rank:
        print(f"  - {name}: {p:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferencia PTB-XL (superclase) con checkpoint elegido")

    parser.add_argument("--checkpoint", type=str,
                        default="C:\\...\\inception1d_best_foldfixed.pt",
                        help="Ruta al checkpoint (.pt)")
    parser.add_argument("--excel_path", type=str,
                        default="C:\\...\\ptbxl_database_test.xlsx",
                        help="Ruta al Excel con columna filename_lr")
    parser.add_argument("--model_name", type=str,
                        default="inception1d",
                        help="Modelo del checkpoint")
    parser.add_argument("--base_dir", type=str,
                        default="C:\\...\\0_ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
                        help="Directorio base donde están los .hea/.dat")

    # Selección de una señal (tal cual aparece en filename_lr)
    parser.add_argument("--signal_name", type=str,
                        default="records100/01000/01386_lr",
                        help="Nombre exacto (columna filename_lr)")

    # Parámetros de filtrado (idénticos a clean_ecg_scipy por defecto)
    parser.add_argument("--lowcut", type=float, default=0.5)
    parser.add_argument("--highcut", type=float, default=40.0)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--powerline", type=float, default=50.0)
    parser.add_argument("--q_notch", type=float, default=30.0)
    parser.add_argument("--no_baseline", action="store_true",
                        help="Si se pasa, NO aplica la eliminación de baseline")

    # Forma del tensor para el modelo
    parser.add_argument("--channels_first", action="store_true",
                        help="Si se pasa, usa (B, C, L) en lugar de (B, L, C)")

    args = parser.parse_args()
    main(args)