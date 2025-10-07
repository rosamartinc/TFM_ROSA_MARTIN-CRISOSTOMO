# TFM – Aplicación de LLM para la interpretación de electrocardiogramas

---

## Descripción del proyecto
Este Trabajo Fin de Máster desarrolla un sistema híbrido **CNN–LLM** para la interpretación automática de electrocardiogramas (ECG).  
El modelo CNN (Inception1D) clasifica señales ECG en cinco superclases diagnósticas, y un modelo de lenguaje (LLM), a elegir entre LLaMA 3 8B, Mistral 7B, GPT-OSS-20B y ClinicalGPT, genera un informe clínico coherente a partir de esas predicciones y de los metadatos del paciente.

El desarrollo se ha realizado íntegramente en **Python**, empleando **entorno local** y herramientas **open source**, garantizando privacidad y reproducibilidad.

Las señales se han obtenido de PTBL-XL ECG Dataset: https://physionet.org/content/ptb-xl/1.0.3/; en concreto, records100 (señales ECG con frecuencia de muestreo igual a 100 Hz).
