from bpe import train_bpe
import json
import os
import time
import tracemalloc
import cProfile

def run_tiny_stories_bpe_experiment():
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    vocab = {vo.decode('latin-1'): id for id, vo in vocab.items()}
    os.makedirs("target", exist_ok=True)
    with open("target/tiny_stories_vocab.json", "w") as f:
        json.dump(vocab, f)
    with open("target/tiny_stories_merges.txt", "w") as f:
        for merge in merges:
            f.write(f"{merge[0].decode('latin-1')} {merge[1].decode('latin-1')}\n")
    return vocab, merges

def run_openwebtext_bpe_experiment():
    input_path = "data/owt_train.txt"
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    vocab = {vo.decode('latin-1'): id for id, vo in vocab.items()}
    os.makedirs("target", exist_ok=True)
    with open("target/openwebtext_vocab.json", "w") as f:
        json.dump(vocab, f)
    with open("target/openwebtext_merges.txt", "w") as f:
        for merge in merges:
            f.write(f"{merge[0].decode('latin-1')} {merge[1].decode('latin-1')}\n")
    return vocab, merges

if __name__ == "__main__":
    t0 = time.time()
    tracemalloc.start()

    cProfile.run('run_tiny_stories_bpe_experiment()', sort='cumulative')

    t1 = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"花费的时间为{t1 - t0:.2f} s\n")
    print(f"消耗的峰值内存为{peak / (2**30):.2f} GB\n")
    # run_openwebtext_bpe_experiment()