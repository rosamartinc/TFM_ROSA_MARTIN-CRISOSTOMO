import argparse
import os
import json
import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
from scipy.signal import butter, filtfilt, iirnotch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def butter_bandpass_filter(x, fs, lowcut=0.5, highcut=40.0, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, x, axis=0)

def notch_filter(x, fs, f0=50.0, q=30.0):
    b, a = iirnotch(w0=f0/(fs/2), Q=q)
    return filtfilt(b, a, x, axis=0)

def moving_average(x, win):
    if win <= 1:
        return x
    # Promedio a lo largo del eje 0
    kernel = np.ones((win,)) / win
    out = np.empty_like(x, dtype=float)
    for ch in range(x.shape[1]):
        out[:, ch] = np.convolve(x[:, ch], kernel, mode='same')
    return out

def moving_median(x, win):
    n, c = x.shape
    out = np.copy(x).astype(float)
    half = win // 2
    for ch in range(c):
        arr = x[:, ch]
        med = np.empty(n)
        for i in range(n):
            i0 = max(0, i - half)
            i1 = min(n, i + half + 1)
            med[i] = np.median(arr[i0:i1])
        out[:, ch] = med
    return out

def remove_baseline_wander(x, fs, enable=True):
    if not enable:
        return x
    # Línea base al estilo IEC: mediana de 200 ms + media de 600 ms
    med_win = max(1, int(0.2 * fs))
    mean_win = max(1, int(0.6 * fs))
    baseline = moving_median(x, med_win)
    baseline = moving_average(baseline, mean_win)
    return x - baseline

def plot_quicklook(t, raw, filt, fs, out_png, max_seconds=10.0, ch=0):
    n = int(min(len(t), max_seconds * fs))
    plt.figure(figsize=(12, 5))
    plt.plot(t[:n], raw[:n, ch], label='Raw')
    plt.plot(t[:n], filt[:n, ch], label='Filtered', alpha=0.9)
    plt.xlabel('Time (s)')
    plt.ylabel(f'Amplitude (ch {ch})')
    plt.title('ECG – Before vs After Filtering (SciPy only)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description='Limpieza de señales')
    ap.add_argument('--excel', required=True, help='Archivo de Excel que contiene una columna con las rutas de los registros WFDB')
    ap.add_argument('--signals-column', default='filename_lr', help='Nombre de la columna con las rutas de señales WFDB')
    ap.add_argument('--data-root', default='.', help='Carpeta raíz')
    ap.add_argument('--output-dir', default='./cleaned_scipy', help='Directorio de salida')
    ap.add_argument('--lowcut', type=float, default=0.5, help='Corte bajo del filtro pasa banda (Hz)')
    ap.add_argument('--highcut', type=float, default=40.0, help='Corte alto del filtro pasa banda (Hz)')
    ap.add_argument('--order', type=int, default=4, help='Orden del filtro Butterworth')
    ap.add_argument('--powerline', type=float, default=50.0, help='Frecuencia de la red eléctrica para el filtro notch (50 o 60)')
    ap.add_argument('--q-notch', type=float, default=30.0, help='Factor de calidad para el filtro notch')
    ap.add_argument('--baseline', action='store_true', help='Habilitar la eliminación de deriva de línea base (mediana + media)')
    ap.add_argument('--limit', type=int, default=None, help='Procesar solo los primeros N registros')
    args = ap.parse_args()

    df = pd.read_excel(args.excel)
    if args.signals_column not in df.columns:
        raise ValueError(f"Column '{args.signals_column}' not found. Available: {list(df.columns)}")

    records = df[args.signals_column].astype(str).tolist()
    if args.limit:
        records = records[:args.limit]

    # Directorios de salida
    
    out_root = Path(args.output_dir)
    out_csv = out_root / 'csv'
    out_npy = out_root / 'npy'
    out_png = out_root / 'png'
    out_meta = out_root / 'meta'
    for p in (out_csv, out_npy, out_png, out_meta):
        ensure_dir(p)

    meta_all = []
    for rec_rel in records:
        rec_base = Path(args.data_root) / rec_rel
        rec_id = rec_base.name
        try:
            sig, fields = wfdb.rdsamp(str(rec_base))
        except Exception as e:
            print(f"[WARN] Skipping {rec_rel}: {e}")
            continue

        fs = float(fields.get('fs', 100.0))
        t = np.arange(sig.shape[0]) / fs

        # Nombres de derivaciones
        
        deriv_names = fields.get('sig_name', None)
        if deriv_names is None or len(deriv_names) != sig.shape[1]:
            deriv_names = [f"Ch {i+1}" for i in range(sig.shape[1])]

        # Filtrado
        
        y = sig.astype(float)
        if args.baseline:
            y = remove_baseline_wander(y, fs, enable=True)
        y = butter_bandpass_filter(y, fs, lowcut=args.lowcut, highcut=args.highcut, order=args.order)
        y = notch_filter(y, fs, f0=args.powerline, q=args.q_notch)

        # Guardado de datos filtrados
        
        np.save(out_npy / f"{rec_id}_filtered.npy", y) # Guardado de npy
        
        # np.savetxt(out_csv / f"{rec_id}_filtered.csv", y, delimiter=',') # Guardado de csv

        # plot_quicklook(t, sig, y, fs, out_png / f"{rec_id}_ch0.png", max_seconds=10.0, ch=0) # PLot de un canal

        # Figura con las 12 derivaciones (Raw vs Filtered)
        n_ch = sig.shape[1]
        plt.figure(figsize=(15, 2.0 * n_ch))
        for i in range(n_ch):
            plt.subplot(n_ch, 1, i+1)
            plt.plot(t, sig[:, i], label="Raw", alpha=0.6)
            plt.plot(t, y[:, i], label="Filtered", linewidth=1)
            ylabel = deriv_names[i] if i < len(deriv_names) else f"Ch {i+1}"
            plt.ylabel(ylabel)
            if i == 0:
                plt.title(f"ECG {rec_id} - Raw vs Filtered ({n_ch} derivaciones)")
            if i < n_ch - 1:
                plt.xticks([])
            plt.legend(loc="upper right", fontsize="small")
        plt.xlabel("Tiempo (s)")
        plt.tight_layout()
        plt.savefig(out_png / f"{rec_id}_all_channels.png", dpi=150)
        plt.close()

        meta = {
            'record': rec_rel,
            'rec_id': rec_id,
            'fs': fs,
            'n_samples': int(sig.shape[0]),
            'n_channels': int(sig.shape[1]),
            'derivations': deriv_names,
            'filters': {
                'baseline': bool(args.baseline),
                'bandpass': {'lowcut': args.lowcut, 'highcut': args.highcut, 'order': args.order},
                'notch': {'f0': args.powerline, 'Q': args.q_notch}
            },
            'outputs': {
                # 'csv': str(out_csv / f"{rec_id}_filtered.csv"),
                'npy': str(out_npy / f"{rec_id}_filtered.npy"),
                'png_all_channels': str(out_png / f"{rec_id}_all_channels.png")
            }
        }
        with open(out_meta / f"{rec_id}.json", 'w') as f:
            json.dump(meta, f, indent=2)
        meta_all.append(meta)
        print(f"[OK] {rec_id} -> saved filtered outputs and plots (SciPy).")

    with open(out_root / 'index.json', 'w') as f:
        json.dump(meta_all, f, indent=2)

if __name__ == '__main__':
    main()
