"""
Manejo de datos para la traducción inglés -> español (Tatoeba).

Formato de los .tsv (sin encabezado, 4 columnas):
    id_en <TAB> oración_en <TAB> id_es <TAB> oración_es
"""

import random
import re
from collections import Counter
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

PAD, SOS, EOS, UNK = "<pad>", "<s>", "</s>", "<unk>"
SPECIALS = [PAD, SOS, EOS, UNK]

Pair = Tuple[str, str]


# --------------------------------------------------------------------------
# Lectura y subconjuntos X / 2X
# --------------------------------------------------------------------------
def load_tatoeba(path: str) -> List[Pair]:
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 4:
                continue
            en, es = cols[1].strip(), cols[3].strip()
            if en and es:
                pairs.append((en, es))
    return pairs


def make_nested_subsets(train_pairs: Sequence[Pair], X: int, seed: int = 1111
                        ) -> Tuple[List[Pair], List[Pair]]:
    """Regresa (subset_X, subset_2X) con subset_X ⊂ subset_2X.

    Se baraja Train UNA sola vez con una semilla fija, se toman los primeros
    2X pares para el experimento 2 y los primeros X de ésos para el experimento 1.
    """
    assert 2 * X <= len(train_pairs), f"2X={2 * X} > |Train|={len(train_pairs)}"
    idx = list(range(len(train_pairs)))
    random.Random(seed).shuffle(idx)
    subset_2x = [train_pairs[i] for i in idx[:2 * X]]
    subset_x = subset_2x[:X]
    return subset_x, subset_2x


# --------------------------------------------------------------------------
# Tokenización
# --------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"\w+(?:'\w+)?|[^\w\s]", re.UNICODE)


def tokenize(text: str, lower: bool = True) -> List[str]:
    """Tokenizador simple: palabras (incluye contracciones tipo don't) y signos de puntuación."""
    if lower:
        text = text.lower()
    return _TOKEN_RE.findall(text)


def detokenize(tokens: Iterable[str]) -> str:
    """Une tokens para producir texto 'natural' antes de calcular BLEU/chrF."""
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,!?;:%)\]}»])", r"\1", text)   # sin espacio antes de signos de cierre
    text = re.sub(r"([¿¡(\[{«])\s+", r"\1", text)         # sin espacio después de signos de apertura
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------
# Vocabulario
# --------------------------------------------------------------------------
class Vocab:
    def __init__(self, min_freq: int = 2, max_size: Optional[int] = None):
        self.min_freq = min_freq
        self.max_size = max_size
        self.itos: List[str] = []
        self.stoi = {}

    def fit(self, token_lists: Iterable[List[str]]) -> "Vocab":
        counter = Counter(t for toks in token_lists for t in toks)
        words = [w for w, c in counter.most_common() if c >= self.min_freq and w not in SPECIALS]
        if self.max_size is not None:
            words = words[: self.max_size - len(SPECIALS)]
        self.itos = SPECIALS + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        return self

    def __len__(self) -> int:
        return len(self.itos)

    pad_id = property(lambda self: self.stoi[PAD])
    sos_id = property(lambda self: self.stoi[SOS])
    eos_id = property(lambda self: self.stoi[EOS])
    unk_id = property(lambda self: self.stoi[UNK])

    def encode(self, tokens: List[str], add_sos: bool = False, add_eos: bool = True) -> List[int]:
        ids = [self.stoi.get(t, self.unk_id) for t in tokens]
        if add_sos:
            ids = [self.sos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: Iterable[int]) -> List[str]:
        out = []
        for i in ids:
            i = int(i)
            if i == self.eos_id:
                break
            if i in (self.pad_id, self.sos_id):
                continue
            out.append(self.itos[i])
        return out


def build_vocabs(pairs: Sequence[Pair], min_freq: int = 2, max_size: Optional[int] = None,
                 lower: bool = True) -> Tuple[Vocab, Vocab]:
    """Vocabularios de origen (en) y destino (es) construidos SÓLO con los pares de entrenamiento."""
    src_vocab = Vocab(min_freq, max_size).fit(tokenize(en, lower) for en, _ in pairs)
    tgt_vocab = Vocab(min_freq, max_size).fit(tokenize(es, lower) for _, es in pairs)
    return src_vocab, tgt_vocab


# --------------------------------------------------------------------------
# Dataset / DataLoader
# --------------------------------------------------------------------------
class TranslationDataset(Dataset):
    """src = w_1..w_n </s>
       tgt = <s> y_1..y_m </s>    (el decoder recibe tgt[:-1] y predice tgt[1:])

    `max_len` filtra pares largos (usar SÓLO en entrenamiento, nunca en Test).
    """

    def __init__(self, pairs: Sequence[Pair], src_vocab: Vocab, tgt_vocab: Vocab,
                 lower: bool = True, max_len: Optional[int] = None):
        self.data = []
        for en, es in pairs:
            s, t = tokenize(en, lower), tokenize(es, lower)
            if max_len is not None and (len(s) > max_len or len(t) > max_len):
                continue
            self.data.append((src_vocab.encode(s), tgt_vocab.encode(t, add_sos=True)))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def make_collate(src_pad: int, tgt_pad: int):
    def collate(batch):
        src_lens = torch.tensor([len(s) for s, _ in batch], dtype=torch.long)
        S, T = int(src_lens.max()), max(len(t) for _, t in batch)
        src = torch.full((len(batch), S), src_pad, dtype=torch.long)
        tgt = torch.full((len(batch), T), tgt_pad, dtype=torch.long)
        for i, (s, t) in enumerate(batch):
            src[i, :len(s)] = torch.tensor(s)
            tgt[i, :len(t)] = torch.tensor(t)
        return src, src_lens, tgt
    return collate


def make_loader(pairs: Sequence[Pair], src_vocab: Vocab, tgt_vocab: Vocab, batch_size: int = 64,
                shuffle: bool = False, lower: bool = True, max_len: Optional[int] = None) -> DataLoader:
    ds = TranslationDataset(pairs, src_vocab, tgt_vocab, lower, max_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=make_collate(src_vocab.pad_id, tgt_vocab.pad_id))
