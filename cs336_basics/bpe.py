import os
from multiprocessing import Pool
from typing import BinaryIO

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def process_chunk(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    start: int,
    end: int
):
    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")

    # process special tokens
    import regex
    split_pat = "|".join(f"(?:{regex.escape(st)})" for st in special_tokens)
    chunks = regex.split(split_pat, text)

    pat = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s""")
    word_freqs = {}
    for chunk in chunks:
        for word in pat.findall(chunk):
            word_bytes = word.encode("utf-8")
            key = tuple(bytes([b]) for b in word_bytes)
            word_freqs[key] = word_freqs.get(key, 0) + 1

    return word_freqs

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    special_tokens = sorted(special_tokens, key=len, reverse=True)

    # read input_path
    with open(input_path, 'rb') as f:
        num_processes = os.cpu_count() or 4
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens[0].encode("utf-8"))
        with Pool(num_processes) as pool:        
            chunk_args = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                chunk_args.append((input_path, special_tokens, start, end))
    
            results = pool.starmap(process_chunk, chunk_args)
    
    word_freqs = {}
    for result in results:
        for word, freq in result.items():
            word_freqs[word] = word_freqs.get(word, 0) + freq

    # init vocab
    vocab = {i: bytes([i]) for i in range(256)}
    for st in special_tokens:
        vocab[len(vocab)] = st.encode("utf-8")

    from collections import Counter
    pair_counts = Counter()
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair_counts[(word[i], word[i + 1])] += freq

    # merge
    merges = []
    num_merges = vocab_size - len(vocab)
    for _ in range(num_merges):
        if not pair_counts:
            break

        freq_pair = max((p for p in pair_counts if pair_counts[p] > 0), key=lambda p: (pair_counts[p], p))
        a, b = freq_pair
        del pair_counts[freq_pair]
        merged = a + b
        merges.append((a, b))
        vocab[len(vocab)] = merged

        to_update = {}
        for word, freq in word_freqs.items():
            if (a, b) not in zip(word, word[1:]):
                continue
                
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    if i > 0:
                        pair_counts[(new_word[-1], word[i])] -= freq
                        pair_counts[(new_word[-1], merged)] += freq

                    new_word.append(merged)

                    if i + 2 < len(word):
                        pair_counts[(word[i + 1], word[i + 2])] -= freq
                        pair_counts[(merged, word[i + 2])] += freq

                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            to_update[word] = (tuple(new_word), freq)
            

        for old_word, (new_word, freq) in to_update.items():
            del word_freqs[old_word]
            word_freqs[new_word] = word_freqs.get(new_word, 0) + freq

    return vocab, merges