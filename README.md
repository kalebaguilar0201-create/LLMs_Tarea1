# Tarea 1 — Attention Recurrent Language Model y Machine Translation

**Curso:** mLLMs / Procesamiento de Lenguaje Natural — CIMAT, Otoño 2026
**Profesor:** Dr. Adrián Pastor López Monroy
**Alumno:** Kaleb Alejandro Aguilar Ávila
**Entrega:** martes 6 de octubre de 2026, 23:59:59

Las instrucciones completas están en [`referencias/mLLMs_Fall_2026_Tarea1_RLM_and_Se2Seq_with_Attention.pdf`](referencias/mLLMs_Fall_2026_Tarea1_RLM_and_Se2Seq_with_Attention.pdf).

---

## Estructura del repositorio

```
.
├── README.md
├── requirements.txt
├── kaleb_alejandro_aguilar_avila.ipynb  # Notebook ÚNICO de entrega (Parte 1 + Parte 2)
│
├── Task1_Language_Modelling/            # Parte 1: código y datos
│   ├── data/
│   │   ├── mex20_train.txt              # tuits MEX-A3T (Train, 5,278 líneas)
│   │   └── mex20_val.txt                # tuits MEX-A3T (Val, 587 líneas)
│   ├── common.py                        # tokenizador, vocabulario compartido, perplejidad, loop de entrenamiento
│   ├── slm.py                           # Punto 2: Statistical LM (n-gramas: Laplace / interpolación)
│   ├── nlm.py                           # Punto 3: Neural LM de Bengio
│   └── rnn_attention_lm.py              # Puntos 1 y 4: LSTM + Self-Attention (Q,K,V) y ablación sin atención
│
├── Task2_Machine_Translation/           # Parte 2: código y datos
│   ├── data/
│   │   ├── tatoeba_train.tsv            # 226,800 pares en–es (80%)
│   │   ├── tatoeba_val.tsv              #  28,350 pares (10%)
│   │   └── tatoeba_test.tsv             #  28,350 pares (10%)
│   ├── mt_data.py                       # carga, subconjuntos anidados X ⊂ 2X, tokenización, vocabularios, DataLoaders
│   └── seq2seq_attention.py             # Punto 1: Encoder–Decoder GRU + atención Q,K,V; entrenamiento; BLEU/chrF
│
└── referencias/                         # PDF de la tarea + notebooks base de la clase
    ├── mLLMs_Fall_2026_Tarea1_RLM_and_Se2Seq_with_Attention.pdf
    ├── Practica_3_SLM.ipynb
    └── Tarea5_NLM.ipynb                 # NLM de Bengio (base de nlm.py)
```

Las carpetas `Task1_Language_Modelling/` y `Task2_Machine_Translation/` sólo contienen **código (`.py`) y datos**. Todos los experimentos se ejecutan desde el notebook único del root, que agrega ambas carpetas al `sys.path`, importa sus módulos y lee los datos desde `<task>/data/`. Los pesos de los modelos entrenados se guardan en `checkpoints/` (ignorado por git).

---

## Notebook de entrega: `kaleb_alejandro_aguilar_avila.ipynb`

Organizado con encabezados markdown en el orden de los puntos del PDF; cada punto termina con una celda de **Observaciones** _(completar)_.

| Sección | Contenido | Módulos usados |
|---|---|---|
| **0. Configuración** | imports, semilla, dispositivo (CUDA / MPS / CPU) | — |
| **Parte 1 — Language Modelling** | | |
| 1.0 Datos y vocabulario | carga de tuits, split train/dev, vocabulario compartido | `common.py` |
| 1.1 Punto 1 | LSTM + Self-Attention (Q,K,V): entrenamiento, PPL y mapa de atención | `rnn_attention_lm.py` |
| 1.2 Punto 2 | SLM de trigramas (Laplace e interpolación, lambdas ajustadas en *dev*) | `slm.py` |
| 1.3 Punto 3 | NLM de Bengio | `nlm.py` |
| 1.4 Punto 4 | ablación **sin atención**, curvas de *dev* y **tabla de perplejidades** | `rnn_attention_lm.py` |
| 1.5 Punto 5 | generación de texto (2 prefijos × modelo) y log-likelihood de oraciones | todos |
| **Parte 2 — Machine Translation** | | |
| 2.0 Datos | carga de Tatoeba y estadísticas de longitud | `mt_data.py` |
| 2.1 Punto 1 | descripción del Seq2Seq y código de `QKVAttention` | `seq2seq_attention.py` |
| 2.2 Punto 2 | experimentos con **X** y **2X** pares, curvas de loss y BLEU en Val | `mt_data.py`, `seq2seq_attention.py` |
| 2.3 Punto 3 | **tabla BLEU / chrF en Test**, ≥ 10 ejemplos (inglés, referencia, X, 2X) y mapa de atención | `seq2seq_attention.py` |
| **3. Comentario general** | LLM usado, dificultades, alucinaciones, recursos y tiempos | — |

---

## Parte 1 — Language Modelling (tuits MEX-A3T)

**Datos:** Train (`mex20_train.txt`) para entrenar y Val (`mex20_val.txt`) para reportar la perplejidad. Para early stopping de los modelos neuronales se aparta un 10 % de Train como *dev*, de modo que Val sólo se usa en la evaluación final.

**Convención común (para que las PPL sean comparables):**
- Mismo tokenizador (`nltk.TweetTokenizer`, minúsculas) y mismo vocabulario (top-5000 de Train, incluye `<pad> <unk> <s> </s>`).
- Cada tuit es independiente; se predicen todas sus palabras más `</s>`; `<s>` nunca se predice.
- `PPL = exp(-(1/N) Σ log P(w_i | historia))`.

| Punto | Pts | Modelo | Archivo | Detalles |
|---|---|---|---|---|
| 1 | 20 | RNN LM con Self-Attention | `Task1_Language_Modelling/rnn_attention_lm.py` | Embedding(256) → LSTM apilada 2×512 **unidireccional** → self-attention causal multi-cabeza (4) con `Q = hW_Q, K = hW_K, V = hW_V`, `softmax(QKᵀ/√d_k + máscara causal)` → `tanh(W_c[h_t; c_t])` → softmax. Adam, dropout 0.4, grad-clip 1.0, early stopping. |
| 2 | 0 | Statistical LM | `Task1_Language_Modelling/slm.py` | Trigramas con suavizado add-k (Laplace) o interpolación lineal (Jelinek-Mercer); lambdas ajustadas sobre *dev*. |
| 3 | 0 | Neural LM (Bengio 2003) | `Task1_Language_Modelling/nlm.py` | Ventana de 3 palabras → embeddings concatenados → `tanh` → dropout → softmax (basado en `referencias/Tarea5_NLM.ipynb`). |
| 4 | 20 | Comparación de PPL | notebook, sección 1.4 | Tabla SLM vs NLM vs LSTM+Att vs **LSTM sin atención** (`use_attention=False`, todo lo demás idéntico). |
| 5 | 10 | Generación de texto | notebook, sección 1.5 | 2 prefijos por modelo (muestreo con temperatura) y log-likelihood de oraciones. |

---

## Parte 2 — Machine Translation inglés → español (Tatoeba)

**Particiones:** Train sólo para entrenar, Val para early stopping / decisiones, Test sólo para la evaluación final.

| Punto | Pts | Contenido | Archivo |
|---|---|---|---|
| 1 | 30 | Seq2Seq **recurrente** (NO Transformer) con atención Q, K, V explícita | `Task2_Machine_Translation/seq2seq_attention.py`; notebook, sección 2.1 |
| 2 | 10 | Experimento 1 con **X** pares y Experimento 2 con **2X** pares (X ⊂ 2X, misma arquitectura y configuración) | `Task2_Machine_Translation/mt_data.py` (`make_nested_subsets`); notebook, sección 2.2 |
| 3 | 10 | BLEU y chrF (sacrebleu) sobre Test, tabla X vs 2X y ≥ 10 ejemplos (inglés, referencia, X, 2X) | notebook, sección 2.3 |

**Arquitectura (`seq2seq_attention.py`):**
- **Encoder:** Embedding(256) → GRU bidireccional 2×512 → `H` (salidas de dimensión 1024) + puente `tanh(W[h_fwd; h_bwd])` para inicializar el decoder.
- **Decoder:** GRU 2×512 con *input feeding* (`[emb(y_{t-1}); z_{t-1}]`).
- **Atención `QKVAttention` (4 cabezas, d_model = 256):**
  - `Q = s_t · W_Q` ← estado oculto del decoder en el paso *t*
  - `K = H · W_K`, `V = H · W_V` ← salidas del encoder (se calculan una vez por oración)
  - `α_t = softmax(Q Kᵀ / √d_k)` con máscara sobre `<pad>`; `c_t = α_t V`
  - `z_t = tanh(W_c [s_t; c_t])`, `P(y_t) = softmax(W_o z_t)`
- **Entrenamiento:** teacher forcing, Adam (lr 1e-3), label smoothing 0.1, grad-clip 1.0, `ReduceLROnPlateau`, early stopping sobre la pérdida de Val. Decodificación greedy.
- **X por defecto = 50,000** (2X = 100,000 ≤ 226,800); se ajusta en la sección 2.2 del notebook según los recursos disponibles. Los vocabularios se construyen sólo con los pares de entrenamiento de cada experimento. Las métricas se calculan en minúsculas (`lowercase=True`) porque el modelo se entrena en minúsculas.

---

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Se recomienda GPU (Google Colab o el cluster de CIMAT), sobre todo para la Parte 2.

## Ejecución

```bash
# desde el root del repositorio
jupyter notebook kaleb_alejandro_aguilar_avila.ipynb
```

En Google Colab:

```python
!git clone https://github.com/kalebaguilar0201-create/LLMs_Tarea1.git
%cd LLMs_Tarea1
!pip install -q sacrebleu nltk
# abrir / ejecutar kaleb_alejandro_aguilar_avila.ipynb desde este directorio
```

## Entrega

Según las instrucciones, se entrega **un notebook ejecutado** con el nombre del alumno: `kaleb_alejandro_aguilar_avila.ipynb`, que ya tiene el nombre y el número de tarea en la primera celda. Las celdas marcadas con _(completar)_ son para las observaciones y comentarios que pide cada punto (LLM usado, dificultades, alucinaciones, recursos, tiempos, etc.).
