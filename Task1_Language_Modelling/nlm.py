"""
Punto 3 — Neural Language Model (Bengio et al., 2003).

Basado en la implementación de la clase (referencias/Tarea5_NLM.ipynb):
ventana de N-1 palabras de contexto -> embeddings concatenados -> capa oculta
con tanh -> dropout -> softmax sobre el vocabulario.

Diferencias respecto al notebook original:
    * usa el `Vocab` compartido (common.py) para que la PPL sea comparable;
    * el padding de inicio usa <s> y cada tuit termina con </s>;
    * perplejidad y entrenamiento vienen de common.py.
"""

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from common import Vocab, sample_from_logits


# --------------------------------------------------------------------------
# Datos: ventanas de n-gramas
# --------------------------------------------------------------------------
def build_ngram_tensors(corpus: Sequence[str], vocab: Vocab, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Para cada tuit:  <s>*(N-1) w_1 ... w_m </s>  ->  (contexto de N-1, siguiente palabra)."""
    X, y = [], []
    for doc in corpus:
        ids = [vocab.sos_id] * (N - 1) + vocab.encode(doc) + [vocab.eos_id]
        for i in range(N - 1, len(ids)):
            X.append(ids[i - N + 1:i])
            y.append(ids[i])
    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def make_loader(corpus: Sequence[str], vocab: Vocab, N: int, batch_size: int = 256,
                shuffle: bool = False) -> DataLoader:
    X, y = build_ngram_tensors(corpus, vocab, N)
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)


# --------------------------------------------------------------------------
# Modelo
# --------------------------------------------------------------------------
class BengioLM(nn.Module):
    def __init__(self, vocab_size: int, N: int = 4, emb_dim: int = 100, hidden_dim: int = 256,
                 dropout: float = 0.3, pad_id: int = 0):
        super().__init__()
        self.N = N
        self.context_size = N - 1
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.fc1 = nn.Linear(emb_dim * self.context_size, hidden_dim)   # h = tanh(W x + b)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)                    # logits = U h + d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N-1)
        e = self.emb(x).view(x.size(0), -1)          # (B, (N-1)*d)
        h = self.drop(torch.tanh(self.fc1(e)))       # (B, d_h)
        return self.fc2(h)                           # (B, V)


# --------------------------------------------------------------------------
# Generación y likelihood
# --------------------------------------------------------------------------
@torch.no_grad()
def generate(model: BengioLM, vocab: Vocab, prefix: str = "", max_len: int = 30,
             temperature: float = 1.0, top_k: Optional[int] = None,
             device: torch.device = torch.device("cpu")) -> str:
    model.eval()
    tokens = vocab.to_tokens(prefix) if prefix else []
    window = [vocab.sos_id] * (model.N - 1) + [vocab.w2id[t] for t in tokens]
    window = window[-(model.N - 1):]
    banned = [vocab.pad_id, vocab.sos_id, vocab.unk_id]
    for _ in range(max_len):
        logits = model(torch.tensor([window], device=device))[0]
        nxt = sample_from_logits(logits, temperature, top_k, banned)
        if nxt == vocab.eos_id:
            break
        tokens.append(vocab.id2w[nxt])
        window = window[1:] + [nxt]
    return " ".join(tokens)


@torch.no_grad()
def sentence_logprob(model: BengioLM, vocab: Vocab, text: str,
                     device: torch.device = torch.device("cpu")) -> float:
    model.eval()
    X, y = build_ngram_tensors([text], vocab, model.N)
    logp = torch.log_softmax(model(X.to(device)), dim=-1)
    return logp[torch.arange(len(y)), y.to(device)].sum().item()
