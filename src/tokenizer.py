"""Lightweight vocabulary tokenizer + tiktoken budget analyzer for email payloads."""
import json
import re
from collections import Counter
import numpy as np
import tiktoken


class EmailTokenizer:
    def __init__(self, max_vocab_size: int = 5000, max_seq_len: int = 64):
        self.max_vocab_size = max_vocab_size
        self.max_seq_len = max_seq_len
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self._tiktoken_enc = tiktoken.get_encoding("cl100k_base")

    @staticmethod
    def clean_text(text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return [tok for tok in text.split() if tok]

    def fit(self, texts: list[str]) -> None:
        counter = Counter()
        for t in texts:
            counter.update(self.clean_text(t))
        for idx, (word, _) in enumerate(counter.most_common(self.max_vocab_size - 2), start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode(self, text: str) -> np.ndarray:
        tokens = self.clean_text(text)
        ids = [self.word2idx.get(tok, 1) for tok in tokens[: self.max_seq_len]]
        if len(ids) < self.max_seq_len:
            ids.extend([0] * (self.max_seq_len - len(ids)))
        return np.array(ids, dtype=np.int64)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.encode(t) for t in texts], axis=0)

    def count_llm_tokens(self, text: str) -> int:
        """Uses tiktoken to measure exact LLM token cost prior to agent escalation."""
        return len(self._tiktoken_enc.encode(text))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "max_vocab_size": self.max_vocab_size,
                    "max_seq_len": self.max_seq_len,
                    "word2idx": self.word2idx,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: str) -> "EmailTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        obj = cls(
            max_vocab_size=payload["max_vocab_size"],
            max_seq_len=payload["max_seq_len"],
        )
        obj.word2idx = payload["word2idx"]
        obj.idx2word = {int(v): k for k, v in obj.word2idx.items()}
        return obj
