"""
Punto 1 — Traductor Seq2Seq inglés -> español con Encoder-Decoder recurrente
(GRU) y atención EXPLÍCITA basada en Query, Key y Value (tipo Transformer).

NO es un Transformer: tanto el encoder como el decoder son recurrentes; lo único
"tipo Transformer" es el bloque de atención (scaled dot-product, multi-cabeza).

    Encoder:  x_1..x_n --Emb--> GRU bidireccional (L capas) --> H = [h_1..h_n]  (B, S, 2h)

    Decoder (paso t):
        s_t = GRU( [emb(y_{t-1}) ; z_{t-1}] , s_{t-1} )           (input feeding)

        Q = s_t   W_Q         <- la QUERY sale del estado del decoder
        K = H     W_K         <- las KEYS salen de las salidas del encoder
        V = H     W_V         <- los VALUES salen de las salidas del encoder

        alpha_t = softmax( Q K^T / sqrt(d_k) , máscara de <pad> )   <- pesos de atención
        c_t     = alpha_t V                                         <- vector de contexto

        z_t = tanh( W_c [s_t ; c_t] )      ->   P(y_t | y_<t, x) = softmax( W_o z_t )
"""

import copy
import math
import random
import time
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from mt_data import Vocab, detokenize, tokenize


# --------------------------------------------------------------------------
# Atención Q, K, V (scaled dot-product, multi-cabeza)
# --------------------------------------------------------------------------
class QKVAttention(nn.Module):
    def __init__(self, query_dim: int, key_dim: int, d_model: int = 256, n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_Q = nn.Linear(query_dim, d_model, bias=False)   # Q = s_t W_Q
        self.W_K = nn.Linear(key_dim, d_model, bias=False)     # K = H   W_K
        self.W_V = nn.Linear(key_dim, d_model, bias=False)     # V = H   W_V
        self.W_O = nn.Linear(d_model, d_model, bias=False)     # concatenación de cabezas
        self.drop = nn.Dropout(dropout)

    def project_keys_values(self, enc_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """K y V sólo dependen del encoder: se calculan UNA vez por oración."""
        B, S, _ = enc_out.shape
        K = self.W_K(enc_out).view(B, S, self.n_heads, self.d_k).transpose(1, 2)   # (B, H, S, d_k)
        V = self.W_V(enc_out).view(B, S, self.n_heads, self.d_k).transpose(1, 2)   # (B, H, S, d_k)
        return K, V

    def forward(self, query: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # query: (B, query_dim) = estado del decoder s_t ; src_mask: (B, S) True en tokens reales
        B = query.size(0)
        Q = self.W_Q(query).view(B, self.n_heads, 1, self.d_k)                    # (B, H, 1, d_k)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)                  # (B, H, 1, S)
        scores = scores.masked_fill(~src_mask[:, None, None, :], float("-inf"))
        alpha = torch.softmax(scores, dim=-1)                                     # pesos de atención
        ctx = self.drop(alpha) @ V                                                # (B, H, 1, d_k)
        ctx = self.W_O(ctx.reshape(B, -1))                                        # (B, d_model)
        return ctx, alpha.squeeze(2)                                              # alpha: (B, H, S)


# --------------------------------------------------------------------------
# Encoder
# --------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, num_layers: int = 2,
                 dropout: float = 0.3, pad_id: int = 0):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.GRU(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                          bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        # estado inicial del decoder a partir de [fwd ; bwd] de cada capa
        self.bridge = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, src: torch.Tensor, src_lens: torch.Tensor):
        e = self.drop(self.emb(src))
        packed = pack_padded_sequence(e, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        out, h_n = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=src.size(1))  # (B, S, 2h)
        B = src.size(0)
        h_n = h_n.view(self.num_layers, 2, B, self.hidden_dim)
        h_n = torch.cat([h_n[:, 0], h_n[:, 1]], dim=-1)                                 # (L, B, 2h)
        dec_init = torch.tanh(self.bridge(h_n))                                         # (L, B, h)
        return out, dec_init


# --------------------------------------------------------------------------
# Decoder con atención
# --------------------------------------------------------------------------
class AttnDecoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, enc_dim: int,
                 num_layers: int = 2, attn_dim: int = 256, n_heads: int = 4, dropout: float = 0.3,
                 pad_id: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.GRU(emb_dim + hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.attention = QKVAttention(query_dim=hidden_dim, key_dim=enc_dim, d_model=attn_dim,
                                      n_heads=n_heads)
        self.combine = nn.Linear(hidden_dim + attn_dim, hidden_dim)   # W_c [s_t ; c_t]
        self.out = nn.Linear(hidden_dim, vocab_size)                  # W_o

    def step(self, y_prev: torch.Tensor, z_prev: torch.Tensor, hidden: torch.Tensor,
             K: torch.Tensor, V: torch.Tensor, src_mask: torch.Tensor):
        """Un paso de decodificación. y_prev: (B,), z_prev: (B, h), hidden: (L, B, h)."""
        e = self.drop(self.emb(y_prev))                                   # (B, emb)
        rnn_in = torch.cat([e, z_prev], dim=-1).unsqueeze(1)              # input feeding
        s, hidden = self.rnn(rnn_in, hidden)
        s = s.squeeze(1)                                                  # s_t: (B, h)  -> Query
        ctx, alpha = self.attention(s, K, V, src_mask)                    # c_t, pesos
        z = torch.tanh(self.combine(torch.cat([s, ctx], dim=-1)))         # z_t
        logits = self.out(self.drop(z))
        return logits, z, hidden, alpha


# --------------------------------------------------------------------------
# Seq2Seq
# --------------------------------------------------------------------------
class Seq2SeqAttention(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, emb_dim: int = 256,
                 hidden_dim: int = 512, num_layers: int = 2, attn_dim: int = 256, n_heads: int = 4,
                 dropout: float = 0.3, src_pad_id: int = 0, tgt_pad_id: int = 0):
        super().__init__()
        self.src_pad_id = src_pad_id
        self.encoder = Encoder(src_vocab_size, emb_dim, hidden_dim, num_layers, dropout, src_pad_id)
        self.decoder = AttnDecoder(tgt_vocab_size, emb_dim, hidden_dim, enc_dim=2 * hidden_dim,
                                   num_layers=num_layers, attn_dim=attn_dim, n_heads=n_heads,
                                   dropout=dropout, pad_id=tgt_pad_id)

    def _encode(self, src, src_lens):
        enc_out, hidden = self.encoder(src, src_lens)
        K, V = self.decoder.attention.project_keys_values(enc_out)
        src_mask = src != self.src_pad_id
        z = enc_out.new_zeros(src.size(0), self.decoder.hidden_dim)
        return K, V, src_mask, hidden, z

    def forward(self, src: torch.Tensor, src_lens: torch.Tensor, tgt_in: torch.Tensor,
                teacher_forcing: float = 1.0) -> torch.Tensor:
        """tgt_in = <s> y_1 ... y_{m-1}   ->   logits (B, T, V) para y_1 ... y_m </s>"""
        K, V, src_mask, hidden, z = self._encode(src, src_lens)
        T = tgt_in.size(1)
        y_prev = tgt_in[:, 0]
        outputs = []
        for t in range(T):
            logits, z, hidden, _ = self.decoder.step(y_prev, z, hidden, K, V, src_mask)
            outputs.append(logits)
            if t + 1 < T:
                use_tf = teacher_forcing >= 1.0 or random.random() < teacher_forcing
                y_prev = tgt_in[:, t + 1] if use_tf else logits.argmax(-1)
        return torch.stack(outputs, dim=1)

    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor, src_lens: torch.Tensor, sos_id: int, eos_id: int,
                      max_len: int = 50, return_attention: bool = False):
        self.eval()
        K, V, src_mask, hidden, z = self._encode(src, src_lens)
        B = src.size(0)
        y_prev = torch.full((B,), sos_id, dtype=torch.long, device=src.device)
        preds, attns = [], []
        finished = torch.zeros(B, dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            logits, z, hidden, alpha = self.decoder.step(y_prev, z, hidden, K, V, src_mask)
            y_prev = logits.argmax(-1)
            preds.append(y_prev)
            attns.append(alpha.mean(1))                                   # promedio de cabezas
            finished |= y_prev == eos_id
            if finished.all():
                break
        preds = torch.stack(preds, dim=1)                                 # (B, T)
        if return_attention:
            return preds, torch.stack(attns, dim=1)                       # (B, T, S)
        return preds


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------
# Entrenamiento
# --------------------------------------------------------------------------
def run_epoch(model, loader, criterion, device, optimizer=None, clip: float = 1.0,
              teacher_forcing: float = 1.0) -> float:
    """Regresa la pérdida promedio por token. Si optimizer es None, sólo evalúa."""
    train = optimizer is not None
    model.train(train)
    total, n_tok = 0.0, 0
    with torch.set_grad_enabled(train):
        for src, src_lens, tgt in loader:
            src, src_lens, tgt = src.to(device), src_lens.to(device), tgt.to(device)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            logits = model(src, src_lens, tgt_in, teacher_forcing if train else 1.0)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
            ntok = (tgt_out != criterion.ignore_index).sum().item()
            total += loss.item() * ntok
            n_tok += ntok
    return total / max(1, n_tok)


def fit(model: Seq2SeqAttention, train_loader, val_loader, tgt_pad_id: int, device: torch.device,
        lr: float = 1e-3, num_epochs: int = 20, patience: int = 3, clip: float = 1.0,
        teacher_forcing: float = 1.0, label_smoothing: float = 0.1,
        save_path: Optional[str] = None, val_bleu_fn: Optional[Callable[[], float]] = None,
        log: Callable[[str], None] = print) -> dict:
    """Entrena con Adam y early stopping sobre la pérdida de VAL (nunca Test).

    Si se pasa `val_bleu_fn` (función sin argumentos que regresa BLEU en una
    muestra de Val), también se registra en el historial.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    train_crit = nn.CrossEntropyLoss(ignore_index=tgt_pad_id, label_smoothing=label_smoothing)
    eval_crit = nn.CrossEntropyLoss(ignore_index=tgt_pad_id)

    history = {"train_loss": [], "val_loss": [], "val_ppl": [], "val_bleu": [], "epoch_time": []}
    best_loss, best_state, bad = float("inf"), None, 0
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, train_crit, device, optimizer, clip, teacher_forcing)
        va = run_epoch(model, val_loader, eval_crit, device)
        scheduler.step(va)
        dt = time.time() - t0
        history["train_loss"].append(tr)
        history["val_loss"].append(va)
        history["val_ppl"].append(math.exp(va))
        history["epoch_time"].append(dt)
        msg = (f"Epoch {epoch:02d} | train loss {tr:.3f} | val loss {va:.3f} "
               f"| val PPL {math.exp(va):7.2f} | lr {optimizer.param_groups[0]['lr']:.1e} | {dt:.0f}s")
        if val_bleu_fn is not None:
            b = val_bleu_fn()
            history["val_bleu"].append(b)
            msg += f" | val BLEU {b:.2f}"
        log(msg)

        if va < best_loss:
            best_loss, bad = va, 0
            best_state = copy.deepcopy(model.state_dict())
            if save_path is not None:
                torch.save(best_state, save_path)
        else:
            bad += 1
            if bad >= patience:
                log(f"Early stopping en epoch {epoch} (mejor val loss = {best_loss:.3f})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val_loss"] = best_loss
    return history


# --------------------------------------------------------------------------
# Traducción y evaluación
# --------------------------------------------------------------------------
@torch.no_grad()
def translate(model: Seq2SeqAttention, sentences: Sequence[str], src_vocab: Vocab, tgt_vocab: Vocab,
              device: torch.device, batch_size: int = 128, max_len: int = 50,
              lower: bool = True) -> List[str]:
    """Traduce una lista de oraciones en inglés (greedy decoding). Conserva el orden."""
    model.eval()
    hyps: List[str] = []
    for i in range(0, len(sentences), batch_size):
        chunk = sentences[i:i + batch_size]
        enc = [src_vocab.encode(tokenize(s, lower)) for s in chunk]
        lens = torch.tensor([len(e) for e in enc], dtype=torch.long)
        src = torch.full((len(enc), int(lens.max())), src_vocab.pad_id, dtype=torch.long)
        for j, e in enumerate(enc):
            src[j, :len(e)] = torch.tensor(e)
        preds = model.greedy_decode(src.to(device), lens.to(device), tgt_vocab.sos_id,
                                    tgt_vocab.eos_id, max_len=max_len)
        hyps.extend(detokenize(tgt_vocab.decode(p)) for p in preds.cpu().tolist())
    return hyps


@torch.no_grad()
def translate_with_attention(model: Seq2SeqAttention, sentence: str, src_vocab: Vocab,
                             tgt_vocab: Vocab, device: torch.device, max_len: int = 50,
                             lower: bool = True):
    """Traduce una oración y regresa (tokens_src, tokens_pred, matriz de atención T x S)."""
    src_toks = tokenize(sentence, lower) + ["</s>"]
    ids = src_vocab.encode(src_toks[:-1])
    src = torch.tensor([ids], device=device)
    lens = torch.tensor([len(ids)], device=device)
    preds, attn = model.greedy_decode(src, lens, tgt_vocab.sos_id, tgt_vocab.eos_id, max_len,
                                      return_attention=True)
    pred_ids = preds[0].tolist()
    T = pred_ids.index(tgt_vocab.eos_id) + 1 if tgt_vocab.eos_id in pred_ids else len(pred_ids)
    out_toks = [tgt_vocab.itos[i] for i in pred_ids[:T]]                     # incluye </s>
    return src_toks, out_toks, attn[0, :T].cpu()


def compute_metrics(hyps: Sequence[str], refs: Sequence[str], lowercase: bool = True) -> dict:
    """BLEU y chrF con sacrebleu (corpus-level)."""
    from sacrebleu.metrics import BLEU, CHRF
    bleu = BLEU(lowercase=lowercase).corpus_score(list(hyps), [list(refs)])
    chrf = CHRF(lowercase=lowercase).corpus_score(list(hyps), [list(refs)])
    return {"BLEU": bleu.score, "chrF": chrf.score, "bleu_obj": bleu, "chrf_obj": chrf}
