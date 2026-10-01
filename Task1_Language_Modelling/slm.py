"""
Punto 2 — Statistical Language Model (modelo de n-gramas).

Modelo de n-gramas (por defecto trigramas) con dos esquemas de suavizado
vistos en clase (Jurafsky, cap. "N-gram Language Models"):

    * 'laplace'       : add-k sobre el n-grama de orden máximo
                        P(w | ctx) = (C(ctx, w) + k) / (C(ctx) + k|V|)

    * 'interpolation' : interpolación lineal (Jelinek-Mercer)
                        P(w | w_{i-2}, w_{i-1}) = l3 P_ML(w | w_{i-2}, w_{i-1})
                                                + l2 P_ML(w | w_{i-1})
                                                + l1 P_add-k(w)
                        (el unigrama se suaviza con add-k para que nunca sea 0)

Usa el mismo `Vocab` que los modelos neuronales (ver common.py), por lo que
la perplejidad es directamente comparable.
"""

import math
import random
from collections import Counter
from typing import List, Optional, Sequence

from common import EOS, PAD, SOS, UNK, Vocab, perplexity_from_nll


class NgramLM:
    def __init__(self, vocab: Vocab, n: int = 3, smoothing: str = "interpolation",
                 k: float = 1.0, lambdas: Optional[Sequence[float]] = None):
        assert smoothing in ("laplace", "interpolation")
        self.vocab = vocab
        self.n = n
        self.smoothing = smoothing
        self.k = k
        if lambdas is None:
            # pesos por defecto (unigrama, bigrama, ..., n-grama)
            lambdas = {1: [1.0], 2: [0.3, 0.7], 3: [0.1, 0.3, 0.6]}.get(n, [1.0 / n] * n)
        assert len(lambdas) == n and abs(sum(lambdas) - 1.0) < 1e-6, "lambdas debe sumar 1"
        self.lambdas = list(lambdas)
        # Palabras que se pueden predecir: todo el vocabulario menos <pad> y <s>
        self.targets = [w for w in vocab.w2id if w not in (PAD, SOS)]
        self.V = len(self.targets)

    # ------------------------------------------------------------------
    def _pad(self, tokens: List[str]) -> List[str]:
        return [SOS] * (self.n - 1) + tokens + [EOS]

    def fit(self, corpus: Sequence[str]) -> "NgramLM":
        """corpus: lista de tuits (texto crudo)."""
        # ngram_counts[m][(w_1..w_m)], ctx_counts[m][(w_1..w_{m-1})]
        self.ngram_counts = [Counter() for _ in range(self.n + 1)]
        self.ctx_counts = [Counter() for _ in range(self.n + 1)]
        for doc in corpus:
            padded = self._pad(self.vocab.to_tokens(doc))
            for i in range(self.n - 1, len(padded)):
                w = padded[i]
                for m in range(1, self.n + 1):
                    ctx = tuple(padded[i - m + 1:i])
                    self.ngram_counts[m][ctx + (w,)] += 1
                    self.ctx_counts[m][ctx] += 1
        return self

    # ------------------------------------------------------------------
    def _mle(self, m: int, ctx: tuple, w: str) -> float:
        d = self.ctx_counts[m][ctx]
        return self.ngram_counts[m][ctx + (w,)] / d if d > 0 else 0.0

    def _addk(self, m: int, ctx: tuple, w: str) -> float:
        c = self.ngram_counts[m][ctx + (w,)]
        d = self.ctx_counts[m][ctx]
        return (c + self.k) / (d + self.k * self.V)

    def prob(self, w: str, history: Sequence[str]) -> float:
        """P(w | historia). `history` ya debe incluir el padding <s>."""
        if w not in self.vocab.w2id:
            w = UNK
        full_ctx = tuple(history[-(self.n - 1):]) if self.n > 1 else ()
        if self.smoothing == "laplace":
            return self._addk(self.n, full_ctx, w)
        p = self.lambdas[0] * self._addk(1, (), w)
        for m in range(2, self.n + 1):
            p += self.lambdas[m - 1] * self._mle(m, full_ctx[len(full_ctx) - (m - 1):], w)
        return p

    # ------------------------------------------------------------------
    def sentence_logprob(self, text: str) -> float:
        padded = self._pad(self.vocab.to_tokens(text))
        return sum(math.log(self.prob(padded[i], padded[:i])) for i in range(self.n - 1, len(padded)))

    def perplexity(self, corpus: Sequence[str]) -> float:
        total_nll, n_tokens = 0.0, 0
        for doc in corpus:
            padded = self._pad(self.vocab.to_tokens(doc))
            for i in range(self.n - 1, len(padded)):
                total_nll -= math.log(self.prob(padded[i], padded[:i]))
                n_tokens += 1
        return perplexity_from_nll(total_nll, n_tokens)

    # ------------------------------------------------------------------
    def generate(self, prefix: str = "", max_len: int = 30, temperature: float = 1.0,
                 seed: Optional[int] = None) -> str:
        """Genera texto muestreando palabra por palabra (sin generar <unk>)."""
        rng = random.Random(seed)
        tokens = [SOS] * (self.n - 1) + (self.vocab.to_tokens(prefix) if prefix else [])
        candidates = [w for w in self.targets if w != UNK]
        for _ in range(max_len):
            probs = [self.prob(w, tokens) ** (1.0 / temperature) for w in candidates]
            w = rng.choices(candidates, weights=probs, k=1)[0]
            if w == EOS:
                break
            tokens.append(w)
        return " ".join(t for t in tokens if t != SOS)
