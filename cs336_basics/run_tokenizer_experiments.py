from tokenizer import Tokenizer

def run_tokenizer_experiments(data_filepath, vocab_filepath, merges_filepath, special_tokens=None):
    docs = []
    current = []
    with open(data_filepath, "r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.strip() in special_tokens:
                if current:
                    docs.append("".join(current))
                    current = []
                if len(docs) == 10:
                    break
            else:
                current.append(line)

    text = "".join(docs)

    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)
    compression_ratio = len(text.encode('utf-8')) / len(tokenizer.encode(text))
    return compression_ratio

if __name__ == "__main__":
    print(run_tokenizer_experiments("data/TinyStoriesV2-GPT4-train.txt", "target/tiny_stories_vocab.json", "target/tiny_stories_merges.txt", ["<|endoftext|>"]))