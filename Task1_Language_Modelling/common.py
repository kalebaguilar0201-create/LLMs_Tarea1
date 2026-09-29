"""
Utilidades compartidas por los tres modelos de lenguaje de la Tarea 1 (Task 1).

Todos los modelos (SLM, NLM de Bengio y RNN con atención) usan EXACTAMENTE
el mismo tokenizador, el mismo vocabulario y la misma convención para contar
tokens, de modo que sus perplejidades sean comparables:

    * Cada tuit se modela de forma independiente.
    * Se predicen todos los tokens del tuit más el token de fin </s>.
    * El token de inicio <s> nunca se predice (sólo se usa como contexto).
    * Las palabras fuera del vocabulario se mapean a <unk>.

    PPL = exp( - (1/N) * sum_i log P(w_i | historia_i) )

donde N es el número total de tokens predichos (palabras + </s>).
"""

import math
import random
from collections import Counter
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from nltk.tokenize import TweetTokenizer

PAD = "<pad>"
UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"
SPECIALS = [PAD, UNK, SOS, EOS]


# --------------------------------------------------------------------------
# Reproducibilidad y dispositivo
# --------------------------------------------------------------------------
def set_seed(seed: int = 1111) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------
def load_tweets(path: str) -> List[str]:
    """Lee un archivo con un tuit por línea (ignora líneas vacías)."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def train_dev_split(corpus: Sequence[str], dev_frac: float = 0.1, seed: int = 1111
                    ) -> Tuple[List[str], List[str]]:
    """Separa una fracción de TRAIN como 'dev' para early stopping.

    Así Val se usa únicamente para reportar la perplejidad final.
    """
    idx = list(range(len(corpus)))
    random.Random(seed).shuffle(idx)
    n_dev = int(len(corpus) * dev_frac)
    dev = [corpus[i] for i in idx[:n_dev]]
    train = [corpus[i] for i in idx[n_dev:]]
    return train, dev


# --------------------------------------------------------------------------
# Vocabulario compartido
# --------------------------------------------------------------------------
class Vocab:
    """Vocabulario a nivel palabra construido sólo con TRAIN.

    Parameters
    ----------
    max_size : tamaño máximo del vocabulario (incluye tokens especiales).
    min_freq : frecuencia mínima para incluir una palabra.
    lower    : si se pasa todo a minúsculas.
    """

    def __init__(self, max_size: int = 5000, min_freq: int = 1, lower: bool = True):
        self.max_size = max_size
        self.min_freq = min_freq
        self.lower = lower
        self._tk = TweetTokenizer(preserve_case=not lower, reduce_len=True)
        self.w2id = {}
        self.id2w = {}

    # ---- tokenización ----
    def tokenize(self, text: str) -> List[str]:
        return self._tk.tokenize(text)

    # ---- construcción ----
    def fit(self, corpus: Iterable[str]) -> "Vocab":
        counter = Counter(tok for doc in corpus for tok in self.tokenize(doc))
        words = [w for w, c in counter.most_common() if c >= self.min_freq and w not in SPECIALS]
        words = words[: self.max_size - len(SPECIALS)]
        itos = SPECIALS + words
        self.w2id = {w: i for i, w in enumerate(itos)}
        self.id2w = {i: w for w, i in self.w2id.items()}
        self.counter = counter
        return self

    def __len__(self) -> int:
        return len(self.w2id)

    def __contains__(self, w: str) -> bool:
        return w in self.w2id

    @property
    def pad_id(self) -> int:
        return self.w2id[PAD]

    @property
    def unk_id(self) -> int:
        return self.w2id[UNK]

    @property
    def sos_id(self) -> int:
        return self.w2id[SOS]

    @property
    def eos_id(self) -> int:
        return self.w2id[EOS]

    # ---- conversión ----
    def to_tokens(self, text: str) -> List[str]:
        """Tokeniza y reemplaza OOV por <unk>."""
        return [t if t in self.w2id else UNK for t in self.tokenize(text)]

    def encode(self, text: str) -> List[int]:
        return [self.w2id.get(t, self.unk_id) for t in self.tokenize(text)]

    def decode(self, ids: Iterable[int], skip_specials: bool = True) -> List[str]:
        out = []
        for i in ids:
            w = self.id2w[int(i)]
            if skip_specials and w in (PAD, SOS):
                continue
            out.append(w)
        return out

    def oov_rate(self, corpus: Iterable[str]) -> float:
        toks = [t for doc in corpus for t in self.tokenize(doc)]
        return sum(t not in self.w2id for t in toks) / max(1, len(toks))


# --------------------------------------------------------------------------
# Métricas
# --------------------------------------------------------------------------
def perplexity_from_nll(total_nll: float, n_tokens: int) -> float:
    """PPL = exp(NLL_total / N)."""
    return math.exp(total_nll / max(1, n_tokens))


@torch.no_grad()
def evaluate_nll(model: nn.Module, loader, device: torch.device, pad_id: int) -> Tuple[float, int]:
    """Suma de -log P(w) sobre todos los tokens no-PAD y número de tokens.

    Funciona para cualquier modelo cuyo forward(x) regrese logits con la
    misma forma que 'y' más la dimensión del vocabulario:
        NLM:  x (B, N-1) -> logits (B, V),     y (B,)
        RNN:  x (B, T)   -> logits (B, T, V),  y (B, T)
    """
    model.eval()
    crit = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        logits = logits[0] if isinstance(logits, tuple) else logits
        total += crit(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
        n += (y != pad_id).sum().item()
    return total, n


def perplexity(model: nn.Module, loader, device: torch.device, pad_id: int) -> float:
    total, n = evaluate_nll(model, loader, device, pad_id)
    return perplexity_from_nll(total, n)


# --------------------------------------------------------------------------
# Loop de entrenamiento genérico (NLM y RNN)
# --------------------------------------------------------------------------
def train_lm(model: nn.Module,
             train_loader,
             dev_loader,
             device: torch.device,
             pad_id: int,
             lr: float = 1e-3,
             weight_decay: float = 0.0,
             num_epochs: int = 30,
             patience: int = 3,
             clip: Optional[float] = 1.0,
             save_path: Optional[str] = None,
             log: Callable[[str], None] = print) -> dict:
    """Entrena con Adam + CrossEntropy, early stopping sobre la PPL de 'dev'.

    Regresa un diccionario con el historial (loss/ppl por época) y deja en
    `model` los pesos del mejor epoch.
    """
    import copy
    import time

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    crit = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")

    history = {"train_ppl": [], "dev_ppl": [], "epoch_time": []}
    best_ppl, best_state, bad_epochs = float("inf"), None, 0

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            logits = logits[0] if isinstance(logits, tuple) else logits
            loss_sum = crit(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            ntok = (y != pad_id).sum()
            (loss_sum / ntok).backward()
            if clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            tot += loss_sum.item()
            n += ntok.item()

        train_ppl = perplexity_from_nll(tot, n)
        dev_ppl = perplexity(model, dev_loader, device, pad_id)
        scheduler.step(dev_ppl)
        dt = time.time() - t0
        history["train_ppl"].append(train_ppl)
        history["dev_ppl"].append(dev_ppl)
        history["epoch_time"].append(dt)
        log(f"Epoch {epoch:02d} | train PPL {train_ppl:9.2f} | dev PPL {dev_ppl:9.2f} "
            f"| lr {optimizer.param_groups[0]['lr']:.1e} | {dt:.1f}s")

        if dev_ppl < best_ppl:
            best_ppl, bad_epochs = dev_ppl, 0
            best_state = copy.deepcopy(model.state_dict())
            if save_path is not None:
                torch.save(best_state, save_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log(f"Early stopping en epoch {epoch} (mejor dev PPL = {best_ppl:.2f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_dev_ppl"] = best_ppl
    return history


# --------------------------------------------------------------------------
# Muestreo
# --------------------------------------------------------------------------
def sample_from_logits(logits: torch.Tensor, temperature: float = 1.0, top_k: Optional[int] = None,
                       banned: Sequence[int] = ()) -> int:
    """Muestrea un índice a partir de logits 1D con temperatura y top-k opcional."""
    logits = logits.detach().float().cpu().clone()
    for b in banned:
        logits[b] = -float("inf")
    if temperature <= 0:
        return int(torch.argmax(logits))
    logits = logits / temperature
    if top_k is not None and top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
        logits[logits < kth] = -float("inf")
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1))
