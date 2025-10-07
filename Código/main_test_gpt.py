import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import wfdb
from fpdf import FPDF
from fpdf.enums import XPos, YPos

from models import build_model
# Importamos SOLO las funciones de filtrado del script clean_ecg_scipy
from clean_ecg_scipy import (
    butter_bandpass_filter,
    notch_filter,
    remove_baseline_wander,
)

DEFAULT_CLASS_NAMES = ["CD", "HYP", "MI", "NORM", "STTC"]

# PTB-XL: 0 = Masculino, 1 = Femenino
SEX_MAP = {0: "Masculino", 1: "Femenino"}


# Utilidades de datos paciente

def is_missing(x):
    return x is None or (isinstance(x, float) and np.isnan(x)) or x == ""


def safe_value(x, fmt=None, missing="No disponible"):
    if is_missing(x) or x == 0:
        return missing
    return fmt(x) if fmt else x


def bmi_and_note(height_cm, weight_kg):
    if is_missing(height_cm) or is_missing(weight_kg):
        return None, None
    try:
        h = float(height_cm)
        w = float(weight_kg)
        if h <= 0 or w <= 0:
            return None, None
        h_m = h if h <= 3.0 else h / 100.0
        bmi = w / (h_m ** 2)
        if bmi < 18.5:
            cat = "Bajo peso"
        elif bmi < 25:
            cat = "Normopeso"
        elif bmi < 30:
            cat = "Sobrepeso"
        else:
            cat = "Obesidad"
        return f"{bmi:.1f}", cat
    except Exception:
        return None, None


def age_group_and_cv_recs(age, sex_str, bmi_val, bmi_cat):
    grupo = "No disponible"
    recs = []

    try:
        a = float(age)
        if a < 18:
            grupo = "Menor de edad"
        elif a < 40:
            grupo = "Adulto joven (18–39)"
        elif a < 65:
            grupo = "Mediana edad (40–64)"
        else:
            grupo = "Adulto mayor (65+)"
    except Exception:
        pass
    
    
    if bmi_val is not None and bmi_cat is not None:
        if bmi_cat in ("Sobrepeso", "Obesidad"):
            recs.insert(0, "Priorizar control ponderal: dieta equilibrada hipocalórica y actividad regular.")
        elif bmi_cat == "Bajo peso":
            recs.insert(0, "Valorar estado nutricional; descartar causas orgánicas si procede.")

    if sex_str == "Femenino":
        recs.append("Considerar riesgos cardiovasculares específicos en mujer (historia obstétrica, menopausia).")

    return grupo, recs


def build_patient_context(row):
    patient_id = safe_value(row.get("patient_id"))
    sex_raw = row.get("sex")
    sex_str = "No disponible"
    if not is_missing(sex_raw):
        try:
            sex_int = int(sex_raw)
            sex_str = SEX_MAP.get(sex_int, "No disponible")
        except Exception:
            sex_str = "No disponible"

    age_str = safe_value(row.get("age"))
    height_str = safe_value(row.get("height"), fmt=lambda v: f"{float(v):.0f} cm")
    weight_str = safe_value(row.get("weight"), fmt=lambda v: f"{float(v):.0f} kg")

    bmi_val, bmi_cat = bmi_and_note(row.get("height"), row.get("weight"))
    if bmi_val is None:
        bmi_line = "IMC: No disponible"
        rec_line = None
    else:
        bmi_line = f"IMC: {bmi_val} ({bmi_cat})"
        rec_line = None
        
    grupo_edad, cv_recs = age_group_and_cv_recs(age_str, sex_str, bmi_val, bmi_cat)

    llm_block = (
        f"Datos del paciente:\n"
        f"- patient_id: {patient_id}\n"
        f"- Sexo: {sex_str}\n"
        f"- Edad: {age_str} ({grupo_edad})\n"
        f"- Altura: {height_str}\n"
        f"- Peso: {weight_str}\n"
        f"- {bmi_line}\n"

    )

    pdf_lines = [
        f"patient_id: {patient_id}",
        f"Sexo: {sex_str}",
        f"Edad: {age_str} ({grupo_edad})",
        f"Altura: {height_str}",
        f"Peso: {weight_str}",
        f"{bmi_line}",
    ]
    if rec_line:
        pdf_lines.append(rec_line)
    pdf_block = "\n".join(pdf_lines)

    return {
        "patient_id": patient_id,
        "sex": sex_str,
        "age": age_str,
        "age_group": grupo_edad,
        "height": height_str,
        "weight": weight_str,
        "bmi_val": bmi_val,
        "bmi_cat": bmi_cat,
        "cv_recs": cv_recs,
        "llm_text": llm_block,
        "pdf_text": pdf_block,
    }


# Procesado señal

def clean_ecg_infer(signal: np.ndarray, fs: float,
                    lowcut: float = 0.5, highcut: float = 40.0,
                    order: int = 4, powerline: float = 50.0,
                    q_notch: float = 30.0, baseline: bool = True) -> np.ndarray:
    y = signal.astype(float)
    if baseline:
        y = remove_baseline_wander(y, fs, enable=True)
    y = butter_bandpass_filter(y, fs, lowcut=lowcut, highcut=highcut, order=order)
    y = notch_filter(y, fs, f0=powerline, q=q_notch)
    return y


def load_signal_wfdb(record_base: Path):
    base = str(record_base).replace(".dat", "").replace(".hea", "")
    sig, fields = wfdb.rdsamp(base)
    fs = float(fields.get("fs", 500.0))
    return sig, fs


# LLM + Construcción prompt 

def build_llm_prompt(signal_name: str, fs: float, pred_name: str, probs_ranked,
                     class_names, patient_ctx_text: str):
    probs_txt = "\n".join([f"- {n}: {p:.4f}" for n, p in probs_ranked])
    prompt = f"""
Eres un cardiólogo que redacta informes breves y claros a partir de la inferencia automática de un ECG de 12 derivaciones (10 s). Toda tu respuesta debe estar redactada únicamente en castellano.

Datos del paciente a considerar:
{patient_ctx_text.strip()}

Datos técnicos:
- Señal: {signal_name}
- Frecuencia de muestreo: {fs:.1f} Hz
- Conjunto de clases: {class_names}
- Predicción del modelo: {pred_name}
- Probabilidades por clase (top-5):
{probs_txt}

Significado de cada clase (diagnóstico): (Importante no hablar de clases usando las siglas, sino su significado)
- ECG normal (NORM)
- Infarto de miocardio (MI)
- Alteraciones ST/T (STTC)
- Trastornos de conducción (CD)
- Hipertrofia (HYP)

Redacta un informe clínico (10–15 líneas), en castellano, con secciones cortas:
- Predicción e implicaciones clínicas (sin emitir diagnóstico definitivo).
- Nivel de confianza (comparando la clase ganadora con las siguientes).
- Pruebas/controles adicionales recomendados y limitaciones del modelo.
- Recomendaciones de estilo de vida/preventivas alineadas con los datos del paciente y con su diagnóstico (clase ganadora). Que las recomendaciones sean reales y específicas.

Usa frases simples, y viñetas si lo ves conveniente.

Recuerda: responde únicamente en castellano, sin traducciones automáticas al inglés.
""".strip()
    return prompt


def call_ollama_llama(model_name: str, prompt: str,
                      endpoint: str = "http://localhost:11434/api/generate",
                      temperature: float = 0.2, max_tokens: int = 600) -> str:
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": 8192,
        },
    }
    r = requests.post(endpoint, json=payload, timeout=10000)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()


# PDF 


def save_report_pdf(text_body: str, out_path: str, title: str, patient_block: str):
    from unicodedata import normalize

    def sanitize_text(s: str | None) -> str:
        if s is None:
            return "No disponible"
        # Normalizamos Unicode para evitar combinaciones raras
        return normalize("NFKC", s)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() != ".pdf":
        out = out.with_suffix(".pdf")

    pdf = FPDF(format="A4", unit="mm")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Fuentes Unicode (Arial de Windows) 
    pdf.add_font("ArialUni", "", r"C:\Windows\Fonts\arial.ttf", uni=True)
    pdf.add_font("ArialUni", "B", r"C:\Windows\Fonts\arialbd.ttf", uni=True)

    usable_w = pdf.w - pdf.l_margin - pdf.r_margin

    # Título 
    pdf.set_font("ArialUni", "B", 18)
    pdf.cell(0, 10, sanitize_text(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Fecha 
    pdf.set_font("ArialUni", "", 10)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(0, 6, sanitize_text(f"Generado: {stamp}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    # Sección: Datos del paciente 
    pdf.set_font("ArialUni", "B", 14)
    pdf.cell(0, 8, "Datos del paciente", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("ArialUni", "", 12)
    for para in sanitize_text(patient_block).split("\n"):
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(usable_w, 6, para)
    pdf.ln(2)

    # Sección: Informe LLM 
    pdf.set_font("ArialUni", "B", 14)
    pdf.cell(0, 8, "Informe clínico (LLM)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("ArialUni", "", 12)

    safe_text = sanitize_text(text_body)
    for para in safe_text.split("\n"):
        if not para.strip():
            pdf.ln(3)
            continue
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(usable_w, 6, para)

    pdf.output(str(out))
    print(f"[INFO] Informe PDF guardado en: {out}")

# Main

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Comenzando el proceso de inferencia para obtener superclase")
    print(f"- Excel: {args.excel_path}")
    print(f"- Base dir: {args.base_dir}")
    print(f"- Señal (filename_lr): {args.signal_name}")
    print(f"- Checkpoint: {args.checkpoint}")

    df = pd.read_excel(args.excel_path)
    if "filename_lr" not in df.columns:
        raise ValueError("La columna 'filename_lr' no existe en el Excel")

    row = df.loc[df["filename_lr"] == args.signal_name]
    if row.empty:
        raise ValueError(f"No se encontró '{args.signal_name}' en {args.excel_path}")

    row = row.iloc[0].to_dict()

    patient_ctx = build_patient_context(row)
    title_text = f"Informe de inferencia ECG — paciente {patient_ctx['patient_id']}"

    wfdb_base = Path(args.base_dir) / row["filename_lr"]

    signal, fs = load_signal_wfdb(wfdb_base)
    signal = clean_ecg_infer(
        signal, fs=fs, lowcut=args.lowcut, highcut=args.highcut,
        order=args.order, powerline=args.powerline, q_notch=args.q_notch,
        baseline=not args.no_baseline,
    )

    sig = signal
    if sig.ndim != 2:
        raise ValueError(f"Se esperaba 2D (L,12) o (12,L); llegó {sig.shape}")

    L, C = sig.shape
    if C == 12:
        sig = np.ascontiguousarray(sig, dtype=np.float32)
    elif L == 12:
        sig = np.ascontiguousarray(sig.T, dtype=np.float32)
    else:
        raise ValueError(f"La señal no tiene 12 canales: {sig.shape}")

    tensor = torch.from_numpy(sig).unsqueeze(0)
    tensor = tensor.permute(0, 2, 1).contiguous().to(device)

    model = build_model(
        args.model_name, n_classes=5, n_leads=12,
        sampling_rate=fs, target_len=1000, dropout=0.3,
    )

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])
    model = model.to(device).eval()

    class_names = DEFAULT_CLASS_NAMES
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1) if logits.shape[1] > 1 else torch.sigmoid(logits)

    probs_np = probs.squeeze(0).cpu().numpy()
    pred_idx = int(probs_np.argmax())
    pred_name = class_names[pred_idx]
    rank = sorted(zip(class_names, probs_np.tolist()), key=lambda x: x[1], reverse=True)

    print("\n=== RESULTADO INFERENCIA SUPERCLASS ===")
    print(f"Señal (filename_lr): {args.signal_name}")
    print(f"Fs: {fs} Hz")
    print(f"Clase predicha (idx): {pred_idx}")
    print(f"Clase predicha (name): {pred_name}")
    print("Probabilidades por clase:")
    for name, p in rank:
        print(f"  - {name}: {p:.6f}")

    if args.make_report:
        try:
            prompt = build_llm_prompt(
                signal_name=args.signal_name,
                fs=fs,
                pred_name=pred_name,
                probs_ranked=rank,
                class_names=DEFAULT_CLASS_NAMES,
                patient_ctx_text=patient_ctx["llm_text"],
            )

            if args.llm_backend == "ollama":
                report_text = call_ollama_llama(model_name=args.llm_model, prompt=prompt)
            else:
                raise ValueError("Solo se soporta 'ollama' en este ejemplo.")

            print("\n--- INFORME (LLM) ---\n")
            print(report_text)
            print("\n--- FIN INFORME ---\n")

            out = Path(args.report_path)
            if out.suffix.lower() != ".pdf":
                out = out.with_suffix(".pdf")

            save_report_pdf(
                text_body=report_text, out_path=str(out),
                title=title_text, patient_block=patient_ctx["pdf_text"],
            )

            meta = {
                "signal_name": args.signal_name,
                "fs": float(fs),
                "pred_idx": int(pred_idx),
                "pred_name": pred_name,
                "probs": {k: float(v) for k, v in zip(DEFAULT_CLASS_NAMES, probs_np.tolist())},
                "rank": [(k, float(v)) for k, v in rank],
                "checkpoint": args.checkpoint,
                "model_name": args.model_name,
                "patient": {
                    "patient_id": patient_ctx["patient_id"],
                    "sex": patient_ctx["sex"],
                    "age": patient_ctx["age"],
                    "age_group": patient_ctx["age_group"],
                    "height": patient_ctx["height"],
                    "weight": patient_ctx["weight"],
                    "bmi": patient_ctx["bmi_val"],
                    "bmi_category": patient_ctx["bmi_cat"],
                },
            }
            meta_path = out.with_suffix(".json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            print(f"[INFO] Metadatos guardados en: {meta_path}")

        except requests.exceptions.RequestException as e:
            print("[WARN] No se pudo contactar con Ollama en http://localhost:11434.")
            print("       ¿Está el servidor arrancado?  ->  ollama serve   y   ollama pull gpt-oss:20b")
            print("Detalle:", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferencia PTB-XL (superclase) con checkpoint elegido")

    parser.add_argument(
        "--checkpoint", type=str,
        default="C:/.../inception1d_best_foldfixed.pt",
    )
    parser.add_argument(
        "--excel_path", type=str,
        default="C:/.../ptbxl_database_test.xlsx",
    )
    parser.add_argument(
        "--model_name", type=str, default="inception1d",
    )
    parser.add_argument(
        "--base_dir", type=str,
        default="C:/.../0_ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
    )
    parser.add_argument(
        "--signal_name", type=str,
        default="records100/02000/02226_lr",
    )

    parser.add_argument("--lowcut", type=float, default=0.5)
    parser.add_argument("--highcut", type=float, default=40.0)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--powerline", type=float, default=50.0)
    parser.add_argument("--q_notch", type=float, default=30.0)
    parser.add_argument("--no_baseline", action="store_true")

    parser.add_argument("--make_report", action="store_true")
    parser.add_argument(
        "--llm_backend", type=str, default="ollama",
        choices=["ollama", "llama_cpp"],
    )
    parser.add_argument(
        "--llm_model", type=str, default="gpt-oss:20b",   
    )
    parser.add_argument(
        "--report_path", type=str,
        default="C:/.../informe_inferencia_ecg.pdf", 
    )

    args = parser.parse_args()
    main(args)
