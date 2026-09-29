"""
Puntos 1 y 4 — Recurrent Language Model con Self-Attention.

Arquitectura (a nivel palabra, un tuit por secuencia):

    x_1..x_T --Embedding--> e_t --LSTM apilada (unidireccional)--> h_t
                                                                    |
               Self-attention causal tipo Transformer (Q, K, V) <---+
                    Q = h W_Q,  K = h W_K,  V = h W_V
                    A = softmax( Q K^T / sqrt(d_k)  + máscara causal )
                    c_t = sum_{j<=t} A_tj V_j            (multi-cabeza)
                                                                    |
               z_t = tanh( W_c [h_t ; c_t] )  --dropout--> softmax(W_o z_t)

Notas de diseño:
    * La LSTM es UNIDIRECCIONAL: una bidireccional "vería" el futuro y no
      sería un modelo de lenguaje válido (P(w_t | w_<t)).
    * La máscara causal impide que la posición t atienda a posiciones j > t.
      Como el padding está a la derecha, la máscara causal también evita que
      los tokens reales atiendan a <pad>.
    * `use_attention=False` elimina el bloque de atención (z_t = tanh(W h_t)),
      con el resto idéntico: es la ablación que pide el punto 4.
"""

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from common import Vocab, sample_from_logits


# --------------------------------------------------------------------------
# Datos: un tuit por secuencia
# --------------------------------------------------------------------------
class TweetLMDataset(Dataset):
    """input  = <s> w_1 ... w_m
       target =     w_1 ... w_m </s>
    Si `max_len` no es None, los tuits largos se parten en trozos de max_len
    (cada trozo conserva el último token del anterior como contexto inicial),
    de modo que NINGÚN token se descarta y la PPL cuenta los mismos tokens
    que el SLM y el NLM."""

    def __init__(self, corpus: Sequence[str], vocab: Vocab, max_len: Optional[int] = None):
        self.samples: List[Tuple[List[int], List[int]]] = []
        for doc in corpus:
            ids = [vocab.sos_id] + vocab.encode(doc) + [vocab.eos_id]
            inp, tgt = ids[:-1], ids[1:]
            if max_len is None or len(inp) <= max_len:
                self.samples.append((inp, tgt))
            else:
                for s in range(0, len(inp), max_len):
                    self.samples.append((inp[s:s + max_len], tgt[s:s + max_len]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def make_collate(pad_id: int):
    def collate(batch):
        T = max(len(inp) for inp, _ in batch)
        x = torch.full((len(batch), T), pad_id, dtype=torch.long)
        y = torch.full((len(batch), T), pad_id, dtype=torch.long)
        for i, (inp, tgt) in enumerate(batch):
            x[i, :len(inp)] = torch.tensor(inp)
            y[i, :len(tgt)] = torch.tensor(tgt)
        return x, y
    return collate


def make_loader(corpus: Sequence[str], vocab: Vocab, batch_size: int = 64, shuffle: bool = False,
                max_len: Optional[int] = None) -> DataLoader:
    ds = TweetLMDataset(corpus, vocab, max_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=make_collate(vocab.pad_id))


# --------------------------------------------------------------------------
# Self-attention causal con Q, K, V explícitos
# --------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model debe ser divisible entre n_heads"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h: (B, T, d_model) -> estados ocultos de la LSTM
        B, T, _ = h.shape

        def split(t):  # (B, T, d) -> (B, heads, T, d_k)
            return t.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        Q = split(self.W_Q(h))
        K = split(self.W_K(h))
        V = split(self.W_V(h))

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)           # (B, H, T, T)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=h.device), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        attn = torch.softmax(scores, dim=-1)                              # pesos de atención
        ctx = self.drop(attn) @ V                                         # (B, H, T, d_k)
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, -1)             # (B, T, d_model)
        return self.W_O(ctx), attn


# --------------------------------------------------------------------------
# Modelo
# --------------------------------------------------------------------------
class RNNAttentionLM(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 256, hidden_dim: int = 512,
                 num_layers: int = 2, n_heads: int = 4, dropout: float = 0.4,
                 rnn_type: str = "lstm", use_attention: bool = True, pad_id: int = 0):
        super().__init__()
        self.use_attention = use_attention
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.emb_drop = nn.Dropout(dropout)
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[rnn_type.lower()]
        self.rnn = rnn_cls(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        if use_attention:
            self.attn = CausalSelfAttention(hidden_dim, n_heads, dropout=0.1)
            self.combine = nn.Linear(2 * hidden_dim, hidden_dim)   # W_c [h_t ; c_t]
        else:
            self.attn = None
            self.combine = nn.Linear(hidden_dim, hidden_dim)       # W_c h_t
        self.out_drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        # x: (B, T)
        e = self.emb_drop(self.emb(x))
        h, _ = self.rnn(e)                                  # (B, T, hidden)
        attn = None
        if self.use_attention:
            c, attn = self.attn(h)
            z = torch.tanh(self.combine(torch.cat([h, c], dim=-1)))
        else:
            z = torch.tanh(self.combine(h))
        logits = self.out(self.out_drop(z))                 # (B, T, V)
        return (logits, attn) if return_attention else logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------
# Generación y likelihood
# --------------------------------------------------------------------------
@torch.no_grad()
def generate(model: RNNAttentionLM, vocab: Vocab, prefix: str = "", max_len: int = 30,
             temperature: float = 1.0, top_k: Optional[int] = None,
             device: torch.device = torch.device("cpu")) -> str:
    """Generación autoregresiva (se recalcula toda la secuencia en cada paso;
    los tuits son cortos, así que es suficientemente rápido)."""
    model.eval()
    tokens = vocab.to_tokens(prefix) if prefix else []
    ids = [vocab.sos_id] + [vocab.w2id[t] for t in tokens]
    banned = [vocab.pad_id, vocab.sos_id, vocab.unk_id]
    for _ in range(max_len):
        logits = model(torch.tensor([ids], device=device))[0, -1]
        nxt = sample_from_logits(logits, temperature, top_k, banned)
        if nxt == vocab.eos_id:
            break
        ids.append(nxt)
        tokens.append(vocab.id2w[nxt])
    return " ".join(tokens)


@torch.no_grad()
def sentence_logprob(model: RNNAttentionLM, vocab: Vocab, text: str,
                     device: torch.device = torch.device("cpu")) -> float:
    model.eval()
    ids = [vocab.sos_id] + vocab.encode(text) + [vocab.eos_id]
    x = torch.tensor([ids[:-1]], device=device)
    y = torch.tensor(ids[1:], device=device)
    logp = torch.log_softmax(model(x)[0], dim=-1)
    return logp[torch.arange(len(y)), y].sum().item()


@torch.no_grad()
def attention_map(model: RNNAttentionLM, vocab: Vocab, text: str,
                  device: torch.device = torch.device("cpu")) -> Tuple[List[str], torch.Tensor]:
    """Regresa los tokens de entrada y la matriz de atención promedio de las cabezas (T x T)."""
    assert model.use_attention, "El modelo no tiene atención"
    model.eval()
    ids = [vocab.sos_id] + vocab.encode(text)
    _, attn = model(torch.tensor([ids], device=device), return_attention=True)
    return vocab.decode(ids, skip_specials=False), attn[0].mean(0).cpu()
