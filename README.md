# Banco de Pruebas Anti-Spoofing — TFG

Aplicación web interactiva para detección de audios deepfake sobre los corpus
**ASVspoof 2019 LA** y **ASVspoof 2021 LA / DF**. Compara extractores DSP
clásicos (RMS, MFCC, LFCC, DWT, CQCC, Fusión) con clasificadores de ML
(Regresión Logística, SVM, XGBoost) y una CNN 2D sobre espectrogramas STFT
(incluida una arquitectura ResNet + Squeeze-and-Excitation con SpecAugment).

Toda la experimentación se gestiona desde la app web (Streamlit); no hay
interfaz de línea de comandos.

## Instalación

```bash
# 1. Clonar y entrar al directorio
git clone <repo-url> && cd TFG

# 2. Entorno virtual
python3 -m venv .venv && source .venv/bin/activate

# 3. Dependencias
pip install -r requirements.txt
```

> El corpus ASVspoof 2019 LA debe descargarse aparte y descomprimirse en
> `data/ASVspoof2019/LA/` (las rutas de los corpus 2021 se configuran en
> `config/config.yaml`). Ver la
> [página oficial del reto](https://datasharing.spsc.tugraz.at/d/8e07dd0b9d/).

## Ejecución

```bash
streamlit run app.py
```

La app abre cuatro páginas:

- **Home** — visión general del benchmark: corpus, metodología y métricas.
- **Signal Explorer** — visualiza el waveform y las representaciones espectrales
  (STFT, entrada CNN, MFCC, LFCC, CQCC) de cualquier audio, o compara una voz
  real frente a un deepfake lado a lado.
- **Run Experiment** — extractor DSP + clasificador clásico sobre 2019 LA, con
  métricas que se acumulan entre ejecuciones y la mejor configuración resaltada.
- **CNN Learning** — entrena la CNN en vivo (curvas de pérdida por época),
  evalúa sobre 2019/2021 y analiza activaciones y predicciones.

## Estructura del proyecto

```
TFG/
├── app.py                    # Entrada de la app web (Streamlit)
├── app_pages/                # Páginas: Home, Signal Explorer, Run Experiment, CNN Learning
├── config/config.yaml        # Parámetros DSP, CNN y rutas de los corpus
├── src/
│   ├── data_loader.py        # Parser de protocolos + Dataset PyTorch
│   ├── features.py           # Extractores DSP (RMS, MFCC, LFCC, DWT, CQCC, STFT)
│   ├── models.py             # Clasificadores clásicos y CNN 2D (+ ResNet/SE)
│   ├── pipeline.py           # Extracción, entrenamiento clásico y CNN
│   ├── metrics.py            # EER y minDCF (Python puro)
│   ├── reporting.py          # Tablas y exportación CSV de resultados
│   └── ui_helpers.py         # CSS, gráficos y componentes compartidos de la UI
├── tests/test_smoke.py       # Tests de humo con señal sintética
└── requirements.txt
```

## Extractores de características

| Nombre | Dimensión | Descripción breve |
|--------|-----------|-------------------|
| RMS Temporal | 2 | Potencia media y varianza por frame. Línea base deliberadamente débil. |
| MFCC | 40 | Cepstrum en escala de Mel. Perceptual; pierde resolución en agudos. |
| LFCC | 40 | Cepstrum con banco lineal. Conserva la banda alta donde viven los artefactos. |
| DWT (db4) | 4 | Energía wavelet multi-resolución; mejor resolución temporal que la STFT. |
| CQCC | 26 | Cepstrum Constant-Q; resolución logarítmica en frecuencia. |
| Fusión | 112 | Concatenación de RMS + MFCC + LFCC + DWT + CQCC (fusión temprana). |

## Clasificadores

| Nombre | Backend |
|--------|---------|
| Regresión Logística (L2) | scikit-learn |
| SVM Lineal (calibrada) | scikit-learn |
| XGBoost | xgboost |
| Todos los clásicos | — |
| CNN 2D / ResNet + SE (espectrograma STFT-dB) | PyTorch (página CNN Learning) |

## Métricas

- **Accuracy** — tasa de acierto con umbral fijo 0.5 sobre p(spoof). Engañosa
  bajo el fuerte desbalance de clases (~1:9), se muestra solo como contexto.
- **EER** — Equal Error Rate: punto donde FAR (deepfakes aceptados como reales)
  = FRR (voces reales rechazadas). Implementado en Python puro.
- **minDCF** — coste de detección NIST (C_miss=1, C_fa=10, P_target=0.05).
- **Tiempo de entrenamiento** y **latencia de inferencia** por audio.

## Tests

```bash
pip install pytest
pytest tests/
```

## Reproducibilidad

La semilla global (`config.yaml → train_params.semilla`) controla los
componentes estocásticos. Cada página de experimentación (Run Experiment,
CNN Learning) expone además un campo de semilla para fijar la ejecución.
