"""
Puntos 1 y 4 — Recurrent Language Model con Self-Attention (bloques apilados).

Arquitectura (a nivel palabra, un tuit por secuencia):

    x_1..x_T --Embedding--> e_t --LSTM o GRU (unidireccional)--> h_t = z^(0)
                                                                    |
     para l = 1..L  (L = n_attn_layers):                            |
          Masked Multi-Head Attention (Vaswani et al., 2017) <------+
               Q = z W_Q,  K = z W_K,  V = z W_V
               Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) + M ) V
               z = LayerNorm( z + Dropout(Attention(z)) )
          Feed-Forward (Linear -> ReLU -> Linear)
               z = LayerNorm( z + Dropout(FFN(z)) )
                                                                    |
               logits = W_o  dropout(z^(L))  --> softmax

Notas de diseño:
    * `rnn_type` = 'lstm' o 'gru'. La RNN es UNIDIRECCIONAL: una bidireccional
      "vería" el futuro y no sería un modelo de lenguaje válido (P(w_t | w_<t)).
    * M es la máscara del decoder del Transformer: M_ij = 0 si j <= i y -inf si
      j > i. Sin ella, la posición i vería la palabra que debe predecir.
      Como el padding está a la derecha, también evita que los tokens reales
      atiendan a <pad>.
    * Cada bloque es una capa clásica de Transformer (post-LN): atención y
      feed-forward con conexión residual + LayerNorm. `ff_mult=0` quita la
      feed-forward (bloque sólo de atención).
    * `use_attention=False` quita la subcapa de atención de cada bloque y deja
      todo lo demás idéntico: es la ablación que pide el punto 4.
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
# Masked Multi-Head Attention (Vaswani et al., 2017 — decoder)
# --------------------------------------------------------------------------
class MaskedMultiHeadAttention(nn.Module):
    """Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) + M ) V

    M es la máscara del decoder: M_ij = 0 si j <= i, -inf si j > i
    (la palabra i sólo puede ver las palabras anteriores y a sí misma)."""

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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, d_model) -> estados de la RNN o salida del bloque anterior
        B, T, D = x.shape
        H, d_k = self.n_heads, self.d_k

        # 1) Proyecciones lineales: Q, K, V
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # 2) Separar en H cabezas: (B, T, D) -> (B, H, T, d_k)
        Q = Q.view(B, T, H, d_k).transpose(1, 2)
        K = K.view(B, T, H, d_k).transpose(1, 2)
        V = V.view(B, T, H, d_k).transpose(1, 2)

        # 3) Scaled dot-product: similitud entre cada query y cada key
        scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)                 # (B, H, T, T)

        # 4) Máscara del decoder: la posición i sólo ve las posiciones j <= i
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        scores = scores.masked_fill(~mask, float("-inf"))

        # 5) Pesos de atención
        attn = torch.softmax(scores, dim=-1)                              # (B, H, T, T)

        # 6) Promedio ponderado de los values
        out = self.drop(attn) @ V                                         # (B, H, T, d_k)

        # 7) Concatenar cabezas y proyección de salida
        out = out.transpose(1, 2).contiguous().view(B, T, D)              # (B, T, D)
        return self.W_O(out), attn


# --------------------------------------------------------------------------
# Bloque tipo Transformer: atención + feed-forward (ReLU), residual y LayerNorm
# --------------------------------------------------------------------------
class AttentionBlock(nn.Module):
    """z = LayerNorm(z + Dropout(Attention(z)))
    z = LayerNorm(z + Dropout(FFN(z)))        FFN = Linear -> ReLU -> Linear

    use_attention=False quita la subcapa de atención (ablación del punto 4).
    ff_mult=0 quita la feed-forward (bloque sólo de atención)."""

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float,
                 use_attention: bool = True, ff_mult: int = 4):
        super().__init__()
        self.use_attention = use_attention
        if use_attention:
            self.attn = MaskedMultiHeadAttention(hidden_dim, n_heads, dropout=0.1)
            self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = None
        if ff_mult > 0:
            self.ff = nn.Sequential(
                nn.Linear(hidden_dim, ff_mult * hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ff_mult * hidden_dim, hidden_dim),
            )
            self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor):
        a = None
        if self.use_attention:
            c, a = self.attn(z)
            z = self.norm1(z + self.drop(c))
        if self.ff is not None:
            z = self.norm2(z + self.drop(self.ff(z)))
        return z, a


# --------------------------------------------------------------------------
# Modelo
# --------------------------------------------------------------------------
class RNNAttentionLM(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 256, hidden_dim: int = 512,
                 num_layers: int = 2, n_heads: int = 4, dropout: float = 0.4,
                 rnn_type: str = "lstm", use_attention: bool = True, pad_id: int = 0,
                 n_attn_layers: int = 10, ff_mult: int = 4):
        super().__init__()
        self.use_attention = use_attention
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.emb_drop = nn.Dropout(dropout)
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[rnn_type.lower()]
        self.rnn = rnn_cls(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        # (atención -> feed-forward) x n_attn_layers
        self.blocks = nn.ModuleList([AttentionBlock(hidden_dim, n_heads, dropout, use_attention, ff_mult)
                                     for _ in range(n_attn_layers)])
        self.out_drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        # x: (B, T)
        z, _ = self.rnn(self.emb_drop(self.emb(x)))          # (B, T, hidden)
        attns = []
        for block in self.blocks:
            z, a = block(z)
            attns.append(a)
        logits = self.out(self.out_drop(z))                  # (B, T, V)
        return (logits, attns) if return_attention else logits


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
                  device: torch.device = torch.device("cpu"),
                  layer: int = -1) -> Tuple[List[str], torch.Tensor]:
    """Tokens de entrada y matriz de atención (T x T) del bloque `layer`
    (por defecto el último), promediando las cabezas."""
    assert model.use_attention, "El modelo no tiene atención"
    model.eval()
    ids = [vocab.sos_id] + vocab.encode(text)
    _, attns = model(torch.tensor([ids], device=device), return_attention=True)
    return vocab.decode(ids, skip_specials=False), attns[layer][0].mean(0).cpu()
