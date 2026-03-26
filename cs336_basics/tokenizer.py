import regex
from collections.abc import Iterable, Iterator

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.reverse_vocab = {v: k for k, v in vocab.items()}

    def _apply_merge(self, word: tuple[bytes], a: bytes, b: bytes):
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                new_word.append(a + b)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)

    def _encode_chunk(self, text: str) -> list[int]:
        ids = []
        PAT = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s""")
        for word in PAT.findall(text):
            word_bytes = word.encode('utf-8')
            key = tuple(bytes([b]) for b in word_bytes)
            for (a, b) in self.merges:
                key = self._apply_merge(key, a, b)
            for token in key:
                ids.append(self.reverse_vocab[token])
        
        return ids

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath) as f:
            raw = json.load(f)
            vocab = {id: token_str.encode('utf-8') for token_str, id in raw.items()}
        
        merges = []
        with open(merges_filepath) as f:
            for line in f:
                parts = line.rstrip().split(" ")
                if len(parts) == 2:
                    token1 = parts[0].encode('utf-8')
                    token2 = parts[1].encode('utf-8')
                    merges.append((token1, token2))
            
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        ids = []

        if self.special_tokens:
            special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            split_pat = "|".join(f"({regex.escape(st)})" for st in special_tokens)
            segments = regex.split(split_pat, text)
            for segment in segments:
                if segment in self.special_tokens:
                    ids.append(self.reverse_vocab[segment.encode()])
                elif segment:
                    ids.extend(self._encode_chunk(segment))
        else:
            ids.extend(self._encode_chunk(text))
        
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for line in iterable:
            for id in self.encode(line):
                yield id

    def decode(self, ids: list[int]) -> str:
        word_bytes = b''.join(self.vocab[i] for i in ids)
        return word_bytes.decode('utf-8', 'replace')