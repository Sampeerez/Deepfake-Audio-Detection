<div align="center">

# Deepfake Audio Detection

### Banco de pruebas interactivo para la detección de voz sintética

*Trabajo de Fin de Grado — Ingeniería Informática*

Compara **17 detectores** —desde front-ends DSP clásicos hasta una red
**wav2vec 2.0** auto-supervisada— sobre los corpus **ASVspoof 2019 / 2021**,
todo desde una aplicación web sin línea de comandos.

<br/>

### **[Abrir la demo en vivo →](https://deepfake-audio-detection-tfg.streamlit.app)**

*Sube tu propio audio y descubre si es voz real o un deepfake en segundos.*

<br/>

[![Streamlit](https://img.shields.io/badge/Demo-Streamlit_Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://deepfake-audio-detection-tfg.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Models-Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)

[![CI](https://github.com/Sampeerez/Deepfake-Audio-Detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Sampeerez/Deepfake-Audio-Detection/actions/workflows/ci.yml)

</div>

---

## Tabla de contenidos

- [¿Qué es esto?](#qué-es-esto)
- [Lo que puedes hacer](#lo-que-puedes-hacer)
- [Resultados destacados](#resultados-destacados)
- [El zoo de detectores](#el-zoo-de-detectores)
- [Corpus de datos](#corpus-de-datos)
- [Métricas](#métricas)
- [Instalación y ejecución](#instalación-y-ejecución)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Pesos de los modelos](#pesos-de-los-modelos)
- [Configuración](#configuración)
- [Tests y CI](#tests-y-ci)
- [Stack tecnológico](#stack-tecnológico)

---

## ¿Qué es esto?

La detección de **deepfakes de audio** (voz sintética generada por TTS o
*voice conversion*) es una de las defensas críticas frente a la suplantación de
identidad por voz. Este proyecto es un **banco de pruebas completo y visual** que
recorre todo el espectro de soluciones al problema:

- desde lo **clásico** — extracción de características DSP a mano (MFCC, LFCC,
  CQCC…) seguida de clasificadores de *machine learning*,
- pasando por **redes convolucionales** que aprenden directamente del
  espectrograma,
- hasta lo **state-of-the-art** — un modelo **wav2vec 2.0** auto-supervisado,
  afinado para anti-spoofing, que digiere la onda de audio en crudo.

Todo se orquesta desde una **aplicación web en Streamlit**: no hay scripts que
ejecutar a mano ni notebooks dispersos. Entrenas, evalúas, comparas y analizas
audios desde el navegador.

> **Hipótesis central del TFG:** las características de **banda alta** (LFCC, CQCC),
> donde se concentran los artefactos del vocoder, superan a las perceptuales
> (MFCC); y los modelos que aprenden la representación de extremo a extremo
> (CNN, wav2vec 2.0) baten a cualquier *pipeline* DSP fijo.

---

## Lo que puedes hacer

La app se organiza en seis páginas:

| Página | Para qué sirve |
|--------|----------------|
| **Home** | Visión general del proyecto: corpus, metodología y métricas de un vistazo. |
| **Signal Explorer** | Visualiza el *waveform* y cada representación espectral (STFT-dB, entrada de la CNN, MFCC, LFCC, CQCC) de cualquier audio, o compara una **voz real frente a un deepfake** lado a lado. |
| **Benchmark** | Tres modos: **Classic models** (DSP × clasificador sobre 2019 LA), **CNN** (entrena la red en vivo con curvas de pérdida por época) y **Full comparison** (evalúa los 17 modelos y construye el *leaderboard*). |
| **Detection Analysis** | **Test an audio** — sube un clip y deja que *todos* los modelos lo puntúen en paralelo, con un veredicto por **fusión ponderada**. **Analyse on a split** — estudia *por qué* un detector logra su EER analizando distribuciones de score, curvas ROC/DET y un umbral de decisión interactivo. |
| **Methodology** | La referencia completa: corpus, front-ends DSP, clasificadores, arquitecturas y métricas en detalle. |
| **Settings** | Tema **Light / Dark Side**, fondo animado, accesibilidad… y algún *easter egg*. |

---

## Resultados destacados

Rendimiento sobre el conjunto **eval de ASVspoof 2019 LA** (menor es mejor):

| Modelo | Tipo | Front-end | EER (%) | minDCF |
|--------|------|-----------|:-------:|:------:|
| **wav2vec 2.0 (SSL)** | Auto-supervisado | Onda en crudo | **4.96** | **0.674** |
| 3-Block CNN (3×3) | CNN 2D | Espectrograma STFT-dB | 9.81 | 0.972 |
| ResNet + SE | CNN 2D + atención | Espectrograma STFT-dB | 10.78 | 0.954 |
| XGBoost · CQCC | ML clásico | CQCC | 13.80 | 0.921 |
| SVM (RBF) · LFCC | ML clásico | LFCC | 15.64 | 0.998 |

> El salto de calidad del modelo **auto-supervisado** sobre todo lo demás (un
> EER ~3× menor que el mejor clásico) es la conclusión más nítida del benchmark.
> Las métricas completas, por modelo y por corpus, viven en
> [`leaderboard.json`](leaderboard.json) y se renderizan en la página *Full
> comparison*.

### Veredicto por fusión ponderada

En **Test an audio**, el dictamen final no sale de un solo modelo sino de una
**fusión tardía ponderada** de las familias más fiables:

```
wav2vec 2.0 (0.40)  +  ResNet + SE (0.20)  +  mejor XGBoost (0.10)
                  └── renormalizados ──┘
```

Si algún miembro no está disponible, los pesos restantes se renormalizan
automáticamente.

---

## El zoo de detectores

**17 modelos** servidos desde el navegador, organizados en tres familias.

### Front-ends DSP (extracción de características)

| Front-end | Dim. | Idea |
|-----------|:----:|------|
| **RMS Temporal** | 2 | Potencia media y varianza por frame. Línea base deliberadamente débil. |
| **MFCC** | 40 | Cepstrum en escala de Mel. Perceptual; pierde resolución en agudos. |
| **LFCC** | 40 | Cepstrum con banco **lineal**. Conserva la banda alta donde viven los artefactos. |
| **DWT (db4)** | 4 | Energía wavelet multirresolución; mejor resolución temporal que la STFT. |
| **CQCC** | 26 | Cepstrum *Constant-Q*; resolución logarítmica en frecuencia. |
| **Fusión** | 112 | Concatenación temprana de RMS + MFCC + LFCC + DWT + CQCC. |

### Clasificadores clásicos

Cada front-end se combina con tres clasificadores → **15 modelos clásicos**:

- **Regresión Logística** (L2)
- **SVM** (kernel RBF + calibración de Platt)
- **XGBoost** (*gradient boosting*)

### Modelos profundos

- **3-Block CNN (3×3)** — CNN 2D de tres bloques convolucionales sobre el
  espectrograma STFT-dB (128 bandas × 300 frames ≈ 9.6 s).
- **ResNet + SE** — cuatro bloques residuales (1→32→64→128→128) con atención de
  canal **Squeeze-and-Excitation** y aumentado de datos **SpecAugment**.
- **wav2vec 2.0 (SSL)** — backbone `Wav2Vec2Model` base de Hugging Face (12
  capas, 768 dim.) afinado para anti-spoofing + cabeza lineal `Linear(768→2)`.
  Recibe la onda a 16 kHz, *mean-pooling* sobre el eje temporal y *temperature
  scaling* (T = 2.0) en la salida. Es de **solo inferencia** (se evalúa, nunca se
  reentrena en la app).

---

## Corpus de datos

| Corpus | Split | Carácter |
|--------|-------|----------|
| **ASVspoof 2019 LA** | train / dev / eval | *Logical Access*: TTS y *voice conversion* de estudio. Base del entrenamiento. |
| **ASVspoof 2021 LA** | eval | Condiciones de **canal telefónico real** (códecs, transmisión). |
| **ASVspoof 2021 DF** | eval | Deepfakes *"in the wild"*, tres particiones, gran diversidad de ataques. |

El audio se distribuye a **16 kHz / 16 bits PCM** (Nyquist = 8 kHz).

> El corpus es voluminoso (varios GB) y **no se incluye en el repositorio**. Para
> ejecutar localmente con datos reales, descarga ASVspoof 2019 LA y descomprímelo
> en `data/ASVspoof2019/LA/` (las rutas de 2021 se definen en
> [`config/config.yaml`](config/config.yaml)). Consulta la
> [página oficial del reto](https://www.asvspoof.org/).
>
> Sin el corpus, la app **sigue siendo plenamente funcional**: se incluye un
> puñado de clips de ejemplo en `samples/` y los splits de evaluación se
> transmiten bajo demanda desde un *dataset* público de Hugging Face — así es como
> funciona la demo en la nube.

---

## Métricas

- **EER** *(Equal Error Rate)* — punto donde la tasa de falsa aceptación
  (deepfakes colados como reales) iguala a la de falso rechazo (voces reales
  bloqueadas). **La métrica principal.** Implementada en Python puro.
- **minDCF** — coste de detección NIST (`C_miss = 1`, `C_fa = 10`,
  `P_target = 0.05`); pondera el coste real de cada tipo de error.
- **Accuracy** — acierto con umbral fijo 0.5 sobre p(spoof). Engañosa bajo el
  fuerte desbalance de clases (~1:9); se muestra solo como contexto.
- **Tiempo de entrenamiento** y **latencia de inferencia** por audio.

---

## Instalación y ejecución

**Requisitos:** Python 3.12 y, opcionalmente, una GPU con CUDA para acelerar el
entrenamiento de la CNN (la inferencia y la demo funcionan a la perfección solo
con CPU).

```bash
# 1. Clonar el repositorio
git clone https://github.com/Sampeerez/Deepfake-Audio-Detection.git
cd Deepfake-Audio-Detection

# 2. Crear el entorno virtual
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Lanzar la app
streamlit run app.py
```

La aplicación se abre en `http://localhost:8501`.

> **Despliegue en Streamlit Cloud (solo CPU):** `requirements.txt` basta para que
> el servidor instale todo. Si el despliegue falla por límite de recursos con la
> rueda de `torch` con CUDA, descomenta las dos líneas indicadas en la cabecera
> de `requirements.txt` para fijar la rueda **CPU** de PyTorch.

---

## Estructura del proyecto

```
Deepfake-Audio-Detection/
├── app.py                      # Entrada de la app: page config, CSS global, navegación, fondo animado
├── app_pages/
│   ├── 0_Home.py               # Landing
│   ├── 1_Signal_Explorer.py    # Visualización y comparación de señales
│   ├── 2_Benchmark.py          # Lanzador de los tres modos de benchmark
│   ├── 3_Detection_Analysis.py # Test an audio + Analyse on a split
│   ├── 4_Methodology.py        # Referencia completa
│   ├── 5_Settings.py           # Tema, fondo, accesibilidad
│   └── modes/                  # _mode_classic / _mode_cnn / _mode_full
├── src/
│   ├── data_loader.py          # Parser de protocolos + Datasets PyTorch (espectrograma y onda)
│   ├── features.py             # Extractores DSP (RMS, MFCC, LFCC, DWT, CQCC, STFT)
│   ├── models.py               # Clasificadores clásicos, CNN 2D, ResNet+SE y Wav2Vec2Classifier
│   ├── pipeline.py             # Extracción, entrenamiento y evaluación
│   ├── metrics.py              # EER y minDCF (Python puro)
│   ├── jobs.py                 # Tareas en segundo plano (sweeps de benchmark)
│   ├── reporting.py            # Tablas y exportación CSV
│   └── ui_helpers.py           # Registro de modelos, descarga HF, CSS, gráficos y componentes UI
├── models/                     # 15 .joblib clásicos + resnet.pth + cnn3x3.pth (wav2vec2.pth se baja de HF)
├── samples/                    # Clips de ejemplo por corpus/subset (la app siempre tiene audio)
├── config/config.yaml          # Parámetros de señal, CNN y rutas de los corpus
├── tests/                      # Suite de tests (metrics, features, models, data_loader, pipeline)
├── leaderboard.json            # Métricas completas por modelo × split (alimenta Full comparison)
├── pytest.ini                  # Configuración de pytest
├── .github/workflows/ci.yml    # Integración continua (GitHub Actions)
├── .streamlit/config.toml      # Tema base y configuración del servidor
└── requirements.txt
```

---

## Pesos de los modelos

El **zoo de modelos va versionado en el repositorio** (`models/`): los 15
clasificadores clásicos (`.joblib`, unos pocos KB cada uno) más las dos CNN
(`resnet.pth`, `cnn3x3.pth`), de modo que se cargan al instante sin descargas.

La **única excepción** es el checkpoint de **wav2vec 2.0** (~469 MB), que supera
el límite duro de 100 MB de GitHub y por eso **no se sube al repositorio**:

- **En local** se lee de `models/wav2vec2.pth`.
- **En la nube** se descarga en el primer arranque desde un repo público de
  Hugging Face declarado en el registro de modelos
  (`Sara1708/deepfake-audio-wav2vec2 → stage2_best.pt`).

Esta excepción está reflejada en [`.gitignore`](.gitignore); todos los demás
modelos sí se versionan.

---

## Configuración

Todos los parámetros físicos de la señal y las rutas viven en
[`config/config.yaml`](config/config.yaml) — el código **nunca** incrusta números
mágicos. Algunos valores clave:

| Parámetro | Valor | Significado |
|-----------|:-----:|-------------|
| `sample_rate` | 16000 Hz | Frecuencia de muestreo (Nyquist = 8 kHz). |
| `n_fft` / `hop_length` | 1024 / 512 | Ventana FFT y salto (50 % de solape). |
| `cnn_input` | 128 × 300 | Bandas de frecuencia × frames temporales (≈ 9.6 s). |
| `epochs` / `batch_size` / `lr` | 20 / 32 / 1e-3 | Entrenamiento de la CNN (Adam). |
| `semilla` | 42 | Semilla global de reproducibilidad. |

---

## Tests y CI

El proyecto incluye una **suite de tests completa** (73 tests, ~10 s) que cubre
toda la lógica de la aplicación con **datos 100 % sintéticos** — no necesita el
corpus ni GPU, así que corre en cualquier máquina y en CI:

```bash
pytest                 # ejecuta toda la suite
pytest tests/test_metrics.py -v    # un módulo concreto, con detalle
```

| Archivo | Qué valida |
|---------|------------|
| `tests/test_metrics.py` | EER y minDCF: separación perfecta/aleatoria, invarianza al orden, guardas de error y sensibilidad a los costes. |
| `tests/test_features.py` | Dimensión exacta de cada front-end DSP, fusión, normalización z-score del espectrograma y el contrato de carga/padding de audio. |
| `tests/test_models.py` | Factory de modelos clásicos (probabilidades calibradas, reproducibilidad) y *forward* de CNN, ResNet+SE y wav2vec 2.0. |
| `tests/test_data_loader.py` | *Parsing* de protocolos 2019/2021, submuestreo estratificado y los dos `Dataset` de PyTorch. |
| `tests/test_pipeline.py` | Extracción de matrices, entrenamiento/evaluación clásica y los *scorers* de inferencia de CNN y onda cruda. |

**Integración continua:** cada *push* y *pull request* a `main` dispara el
workflow de **GitHub Actions** ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)),
que instala las dependencias (PyTorch CPU) en Python 3.11 y 3.12 y ejecuta la
suite completa. El estado se refleja en el *badge* de CI al inicio de este README.

> **Reproducibilidad:** la **semilla global** (`config.yaml → train_params.semilla`)
> controla todos los componentes estocásticos; además, cada página de
> experimentación expone su propio campo de semilla para fijar la ejecución.

---

## Stack tecnológico

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=flat-square)
![Transformers](https://img.shields.io/badge/🤗_Transformers-FFD21E?style=flat-square)
![librosa](https://img.shields.io/badge/librosa-4D02A2?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Altair](https://img.shields.io/badge/Altair-1F77B4?style=flat-square)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)

</div>

- **DSP & audio:** librosa, SciPy, PyWavelets, soundfile
- **Machine learning:** scikit-learn, XGBoost, PyTorch, Hugging Face Transformers
- **App & visualización:** Streamlit, Altair, Matplotlib, pandas

---

<div align="center">

**[· Probar la demo en vivo ·](https://deepfake-audio-detection-tfg.streamlit.app)**

Trabajo de Fin de Grado · Ingeniería Informática

</div>
