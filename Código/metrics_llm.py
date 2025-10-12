#pip install --upgrade --no-cache-dir rouge-score bert-score sentence-transformers torch pandas openpyxl 

import argparse
import os
import re
import statistics
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
from rouge_score import rouge_scorer  # ROUGE
from bert_score import score as bert_score  # BERTScore
import pandas as pd 

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def split_sentences_es(text: str) -> List[str]:
    text = text.replace("\n", " ")
    abbr = r"(Sr|Sra|Dr|Dra|Lic|Ing|Av|Ud|Uds|p\.ej|etc|No)\."
    text = re.sub(abbr, lambda m: m.group(0).replace(".", "<ABBR_DOT>"), text, flags=re.IGNORECASE)
    parts = re.split(r"(?<=[\.!?;])\s+", text)
    return [p.replace("<ABBR_DOT>", ".").strip() for p in parts if p.strip()]

def tokenize_words(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())

def count_syllables_es(word: str) -> int:
    w = re.sub(r"[^a-záéíóúüñ]", "", word.lower())
    if not w:
        return 0
    groups = re.findall(r"[aeiouáéíóúü]+", w)
    return max(1, len(groups))

def fernandez_huerta_index(text: str) -> float:
    sentences = split_sentences_es(text)
    words = tokenize_words(text)
    if not words or not sentences:
        return 0.0
    syllables = sum(count_syllables_es(w) for w in words)
    words_count = len(words)
    sent_count = len(sentences)
    P = (syllables / words_count) * 100.0
    F = words_count / sent_count
    return 206.84 - 0.60 * P - 1.02 * F

def readability_band(ifh: float) -> str:
    if ifh >= 90: return "Muy fácil"
    if ifh >= 80: return "Fácil"
    if ifh >= 70: return "Bastante fácil"
    if ifh >= 60: return "Normal"
    if ifh >= 50: return "Algo difícil"
    if ifh >= 30: return "Difícil"
    return "Muy difícil"

# Métricas básicas

@dataclass
class BasicMetrics:
    char_count: int
    word_count: int
    sentence_count: int
    avg_sentence_len_words: float
    std_sentence_len_words: float
    type_token_ratio: float
    fernandez_huerta: float
    readability_band: str

def compute_basic_metrics(text: str) -> BasicMetrics:
    sentences = split_sentences_es(text)
    words = tokenize_words(text)
    word_count = len(words)
    sentence_lens = [len(tokenize_words(s)) for s in sentences]
    avg_len = statistics.mean(sentence_lens) if sentence_lens else 0.0
    std_len = statistics.pstdev(sentence_lens) if len(sentence_lens) > 1 else 0.0
    ttr = (len(set(words)) / word_count) if word_count else 0.0
    ifh = fernandez_huerta_index(text)
    band = readability_band(ifh)
    return BasicMetrics(
        char_count=len(text),
        word_count=word_count,
        sentence_count=len(sentences),
        avg_sentence_len_words=round(avg_len, 3),
        std_sentence_len_words=round(std_len, 3),
        type_token_ratio=round(ttr, 4),
        fernandez_huerta=round(ifh, 2),
        readability_band=band,
    )

# ROUGE y BERTScore 

def try_compute_rouge(system_text: str, reference_text: str) -> Optional[Dict[str, Dict[str, float]]]:
    """Calcula las métricas ROUGE (1, 2 y L) entre dos textos."""
    try:
        from rouge_score import rouge_scorer
    except Exception:
        return None

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_text, system_text)

    out = {
        k: {
            "precision": round(v.precision, 4),
            "recall": round(v.recall, 4),
            "f1": round(v.fmeasure, 4)
        }
        for k, v in scores.items()
    }

    return out


def try_compute_bertscore(system_text: str, reference_text: str) -> Optional[Dict[str, float]]:
    """
    Calcula BERTScore (Precision, Recall, F1) entre el texto generado y el de referencia.
    Usa un modelo multilingüe robusto para español (LaBSE).
    """
    try:
        from bert_score import score as bert_score
    except Exception:
        return None

    try:
        # LaBSE, porque el texto es en español
        P, R, F1 = bert_score(
            [system_text],
            [reference_text],
            model_type="sentence-transformers/LaBSE",
            lang="es",
            rescale_with_baseline=False 
        )
    except Exception:
        # Modelo base si LaBSE no está disponible
        P, R, F1 = bert_score(
            [system_text],
            [reference_text],
            model_type="bert-base-multilingual-cased",
            lang="es",
            rescale_with_baseline=False
        )

    return {
        "precision": round(float(P.mean()), 4),
        "recall": round(float(R.mean()), 4),
        "f1": round(float(F1.mean()), 4)
    }

    return {
        "precision": round(float(P.mean()), 4),
        "recall": round(float(R.mean()), 4),
        "f1": round(float(F1.mean()), 4)
    }


# Chequeo de facts pasados como argumento

@dataclass
class FactChecks:
    age_number_present: Optional[bool] = None
    disease_present: Optional[bool] = None
    custom_fields_present: Dict[str, bool] = None

"""
def compute_fact_checks(text: str, edad: Optional[int] = None, diagnostico: Optional[str] = None, **extra) -> FactChecks:
    text_low = text.lower()
    age_ok = str(int(edad)) in text_low if edad is not None else None
    disease_ok = diagnostico.lower() in text_low if diagnostico else None
    custom = {k: str(v).lower() in text_low for k, v in extra.items() if v is not None}
    return FactChecks(age_number_present=age_ok, disease_present=disease_ok, custom_fields_present=custom)
"""
def compute_fact_checks(text: str, edad: Optional[int] = None, diagnostico: Optional[Any] = None, **extra) -> FactChecks:
    text_low = text.lower()
    age_ok = str(int(edad)) in text_low if edad is not None else None

    # Normaliza diagnostico a lista de strings
    if diagnostico is None:
        diag_list = []
    elif isinstance(diagnostico, str):
        # permite escribir tanto "HYP,STTC" como "HYP STTC"
        diag_list = [d for d in re.split(r"[,\s]+", diagnostico) if d]
    else:
        diag_list = list(diagnostico)

    diag_list_low = [d.lower() for d in diag_list]

    # Presencia por etiqueta (en caso de varias etiquetas tienen que estar todas para cosiderar Verdadero)
    per_label = {f"diagnostico_{d}_present": (d.lower() in text_low) for d in diag_list}

    if diag_list_low:
        # AND: TODAS TIENEN QUE APARECER
        all_ok = all(lbl in text_low for lbl in diag_list_low)
    else:
        all_ok = None

    # Campos extra 
    custom = {k: (str(v).lower() in text_low) for k, v in extra.items() if v is not None}
    custom.update(per_label)
    custom["diagnostico_all_present"] = all_ok

    return FactChecks(age_number_present=age_ok, disease_present=all_ok, custom_fields_present=custom)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def flatten_for_excel(basic: BasicMetrics, rouge: Optional[Dict[str, Dict[str, float]]], bert: Optional[Dict[str, float]], fact: FactChecks, meta: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    row.update(asdict(basic))
    if rouge:
        for k, vals in rouge.items():
            for metric, val in vals.items():
                row[f"{k}_{metric}"] = val
    if bert:
        for metric, val in bert.items():
            row[f"bertscore_{metric}"] = val
    row["fact_age_present"] = fact.age_number_present
    row["fact_disease_present"] = fact.disease_present
    for k, v in (fact.custom_fields_present or {}).items():
        row[f"fact_{k}_present"] = v
    row.update(meta)
    return row


def main():
    parser = argparse.ArgumentParser(description="Evaluación cuantitativa de informes clínicos generados por LLM.")
    parser.add_argument("--text", type=str, help="Texto del informe a evaluar", default=""" Basado en el análisis de los datos del ECG del paciente 10684, una mujer adulta de 71 años, el
 modelo predice un diagnóstico de hipertrofia (HYP). Sin embargo, es importante destacar que este
 no es un diagnóstico definitivo. El modelo tiene un nivel alto de confianza en esta predicción, con
 una probabilidad de 0,3925. Esto es significativamente más alto que las probabilidades para otros
 posibles diagnósticos, como alteraciones ST/T (STTC) a 0,3437, infarto miocárdico (MI) a 0,1952,
 trastornos de conducción (CD) a 0,0311 y ECG normal (NORM) a 0,0375.
 La confianza del modelo en la predicción de hipertrofia se ve respaldada por la edad y el género del
 paciente. Sin embargo, es importante destacar que esta predicción no debe considerarse definitiva
 sin una evaluación clínica adicional. No se proporcionaron la altura y el peso del paciente, lo que
 podría afectar la precisión de la predicción.
 Se pueden recomendar pruebas o controles adicionales basadas en los resultados de este análisis.
 Por ejemplo, mediciones de presión arterial podrían ayudar a confirmar un diagnóstico de
 hipertrofia. Sin embargo, es importante destacar que las predicciones del modelo se basan solo en
 datos del ECG y no tienen en cuenta otros factores que podrían influir en el estado de salud del
 paciente.
 En cuanto a recomendaciones de estilo de vida, dado el diagnóstico predicho de hipertrofia, el
 paciente debería considerar adoptar una dieta saludable rica en frutas y verduras para ayudar a
 manejar su condición. La actividad física regular también se recomienda para mantener la salud
 cardiovascular. Sin embargo, estas recomendaciones deben discutirse con un profesional de la
 salud para asegurarse de que sean adecuadas para las necesidades específicas y circunstancias
 individuales del paciente.
 En conclusión, aunque el modelo prediga hipertrofia como el diagnóstico más probable basado en
 los datos del ECG, es importante destacar que esta predicción no debe considerarse definitiva sin
 una evaluación clínica adicional. Se pueden recomendar pruebas o controles adicionales para
 confirmar el diagnóstico. Las recomendaciones de estilo de vida, como una dieta saludable y
 actividad física regular, podrían ayudar a manejar la hipertrofia si se confirma por un profesional de
 la salud.
 Este informe tiene como objetivo proporcionar una visión general general del estado del paciente
 basado en los datos del ECG. No""")
    parser.add_argument("--reference", type=str, help="Texto de referencia", default="Ritmo regular, no se encuentran ondas P, eje eléctrico izquierdo, hipertrofia del ventrículo izquierdo")
    parser.add_argument("--edad", type=int, help="Edad del paciente", default=71)
    # parser.add_argument("--diagnostico", type=str, help="Diagnóstico", default="MI")
    parser.add_argument("--diagnostico", type=str, help="Diagnóstico", default=["HYP", "STTC"]) #fact_disease_present será True solo si todas las etiquetas (p.ej. HYP y STTC); se sacarán las columnas: fact_diagnostico_HYP_present, fact_diagnostico_STTC_present y fact_diagnostico_all_present 
    parser.add_argument("--sexo", type=str, nargs="+", help="Sexo del paciente", default="Femenino")
    parser.add_argument("--otros", nargs="*", help="Otros campos opcionales como clave=valor", default=[])
    parser.add_argument("--out-dir", type=str, help="Carpeta destino para el Excel", default="C:\\...\\metrics_llm")
    parser.add_argument("--excel-name", type=str, help="Nombre del Excel de salida", default="metrics_15787_HYP_STTC_clinical.xlsx")

    args = parser.parse_args()

    text = normalize_text(args.text)
    reference = normalize_text(args.reference) if args.reference else None

    extra_fields = {}
    if args.otros:
        for kv in args.otros:
            if "=" in kv:
                k, v = kv.split("=", 1)
                extra_fields[k] = v

    basic = compute_basic_metrics(text)
    rouge = try_compute_rouge(text, reference) if reference else None
    bert = try_compute_bertscore(text, reference) if reference else None
    fact = compute_fact_checks(text, edad=args.edad, diagnostico=args.diagnostico, sexo=args.sexo, **extra_fields)

    ensure_dir(args.out_dir)
    excel_path = os.path.join(args.out_dir, args.excel_name)

    
    """
    row = flatten_for_excel(
        basic, rouge, bert, fact,
        meta={"edad": args.edad, "diagnostico": args.diagnostico, "sexo": args.sexo, **extra_fields}
    )
    """
    
    row = flatten_for_excel(
        basic, rouge, bert, fact,
        meta={
            "edad": args.edad,
            "diagnostico": "|".join(args.diagnostico) if isinstance(args.diagnostico, list) else str(args.diagnostico),
            "sexo": args.sexo,
            **extra_fields
        }
    )
    
    
    new_df = pd.DataFrame([row])

    if os.path.exists(excel_path):
        try:
            old_df = pd.read_excel(excel_path)
            out_df = pd.concat([old_df, new_df], ignore_index=True)
        except Exception:
            out_df = new_df
    else:
        out_df = new_df

    out_df.to_excel(excel_path, index=False)

    print(f"\nResultados guardados en Excel: {excel_path}\n")

if __name__ == "__main__":
    main()
