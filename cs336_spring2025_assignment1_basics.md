# CS336 Assignment 1 (basics): Building a Transformer LM

Version 1.0.6

CS336 Staff

Spring 2025

# 1 Assignment Overview

In this assignment, you will build all the components needed to train a standard Transformer language model(LM) from scratch and train some models.

# What you will implement

1. Byte-pair encoding (BPE) tokenizer (§2)

2. Transformer language model (LM) (§3)

3. The cross-entropy loss function and the AdamW optimizer (§4)

4. The training loop, with support for serializing and loading model and optimizer state (§5)

# What you will run

1. Train a BPE tokenizer on the TinyStories dataset.

2. Run your trained tokenizer on the dataset to convert it into a sequence of integer IDs.

3. Train a Transformer LM on the TinyStories dataset.

4. Generate samples and evaluate perplexity using the trained Transformer LM.

5. Train models on OpenWebText and submit your attained perplexities to a leaderboard.

What you can use We expect you to build these components from scratch. In particular, you may notuse any definitions from torch.nn, torch.nn.functional, or torch.optim except for the following:

• torch.nn.Parameter

• Container classes in torch.nn (e.g., Module, ModuleList, Sequential, etc.)1

• The torch.optim.Optimizer base class

You may use any other PyTorch definitions. If you would like to use a function or class and are notsure whether it is permitted, feel free to ask on Slack. When in doubt, consider if using it compromises the“from-scratch” ethos of the assignment.

Statement on AI tools Prompting LLMs such as ChatGPT is permitted for low-level programmingquestions or high-level conceptual questions about language models, but using it directly to solve the problemis prohibited.

We strongly encourage you to disable AI autocomplete (e.g., Cursor Tab, GitHub CoPilot) in your IDEwhen completing assignments (though non-AI autocomplete, e.g., autocompleting function names is totallyfine). We have found that AI autocomplete makes it much harder to engage deeply with the content.

What the code looks like All the assignment code as well as this writeup are available on GitHub at:

github.com/stanford-cs336/assignment1-basics

Please git clone the repository. If there are any updates, we will notify you so you can git pull to getthe latest.

1. cs336_basics/*: This is where you write your code. Note that there’s no code in here—you can dowhatever you want from scratch!

2. adapters.py: There is a set of functionality that your code must have. For each piece offunctionality (e.g., scaled dot product attention), fill out its implementation (e.g.,run_scaled_dot_product_attention) by simply invoking your code. Note: your changes toadapters.py should not contain any substantive logic; this is glue code.

3. test_*.py: This contains all the tests that you must pass (e.g.,test_scaled_dot_product_attention), which will invoke the hooks defined in adapters.py. Don’tedit the test files.

How to submit You will submit the following files to Gradescope:

• writeup.pdf: Answer all the written questions. Please typeset your responses.

• code.zip: Contains all the code you’ve written.

To submit to the leaderboard, submit a PR to:

github.com/stanford-cs336/assignment1-basics-leaderboard

See the README.md in the leaderboard repository for detailed submission instructions.

Where to get datasets This assignment will use two pre-processed datasets: TinyStories [Eldan and Li,2023] and OpenWebText [Gokaslan et al., 2019]. Both datasets are single, large plaintext files. If you aredoing the assignment with the class, you can find these files at /data of any non-head node machine.

If you are following along at home, you can download these files with the commands inside the README.md.

# Low-Resource/Downscaling Tip: Init

Throughout the course’s assignment handouts, we will give advice for working through parts of theassignment with fewer or no GPU resources. For example, we will sometimes suggest downscalingyour dataset or model size, or explain how to run training code on a MacOS integrated GPU or CPU.You’ll find these “low-resource tips” in a blue box (like this one). Even if you are an enrolled Stanfordstudent with access to the course machines, these tips may help you iterate faster and save time, so werecommend you to read them!

# Low-Resource/Downscaling Tip: Assignment 1 on Apple Silicon or CPU

With the staff solution code, we can train an LM to generate reasonably fluent text on an Apple M3Max chip with 36 GB RAM, in under 5 minutes on Metal GPU (MPS) and about 30 minutes using theCPU. If these words don’t mean much to you, don’t worry! Just know that if you have a reasonablyup-to-date laptop and your implementation is correct and efficient, you will be able to train a smallLM that generates simple children’s stories with decent fluency.

Later in the assignment, we will explain what changes to make if you are on CPU or MPS.

# 2 Byte-Pair Encoding (BPE) Tokenizer

In the first part of the assignment, we will train and implement a byte-level byte-pair encoding (BPE)tokenizer [Sennrich et al., 2016, Wang et al., 2019]. In particular, we will represent arbitrary (Unicode)strings as a sequence of bytes and train our BPE tokenizer on this byte sequence. Later, we will use thistokenizer to encode text (a string) into tokens (a sequence of integers) for language modeling.

# 2.1 The Unicode Standard

Unicode is a text encoding standard that maps characters to integer code points. As of Unicode 16.0 (releasedin September 2024), the standard defines 154,998 characters across 168 scripts. For example, the character“s” has the code point 115 (typically notated as U+0073, where U+ is a conventional prefix and 0073 is 115 inhexadecimal), and the character “牛” has the code point 29275. In Python, you can use the ord() functionto convert a single Unicode character into its integer representation. The chr() function converts an integerUnicode code point into a string with the corresponding character.

```txt
>>>ord('牛') 29275   
>>>chr(29275) '牛'
```

# Problem (unicode1): Understanding Unicode (1 point)

(a) What Unicode character does chr(0) return?

Deliverable: A one-sentence response.

(b) How does this character’s string representation (__repr__()) differ from its printed representa-tion?

Deliverable: A one-sentence response.

(c) What happens when this character occurs in text? It may be helpful to play around with thefollowing in your Python interpreter and see if it matches your expectations:

>>>chr(0)   
>>>print(chr(0))   
>>> "this is a test"  $^+$  chr(0)  $^+$  "string"   
>>> print("this is a test"  $^+$  chr(0)  $^+$  "string")

Deliverable: A one-sentence response.

# 2.2 Unicode Encodings

While the Unicode standard defines a mapping from characters to code points (integers), it’s impractical totrain tokenizers directly on Unicode codepoints, since the vocabulary would be prohibitively large (around150K items) and sparse (since many characters are quite rare). Instead, we’ll use a Unicode encoding, whichconverts a Unicode character into a sequence of bytes. The Unicode standard itself defines three encodings:UTF-8, UTF-16, and UTF-32, with UTF-8 being the dominant encoding for the Internet (more than 98%of all webpages).

To encode a Unicode string into UTF-8, we can use the encode() function in Python. To access theunderlying byte values for a Python bytes object, we can iterate over it (e.g., call list()). Finally, we canuse the decode() function to decode a UTF-8 byte string into a Unicode string.

```python
>>> test_string = "hello! 你怎么是"
>>> utf8-encoded = test_string.encode("utf-8")
>>> print utf8-encoded)
b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\xa1\xe3\xa81\xaf!"
>>> print(typeutf8-encoded))
<class 'bytes">
>>> # Get the byte values for the encoded string (integers from 0 to 255).
>>> list utf8-encoded)
[104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129, 161, 227, 129, 175, 33]
>>> # One byte does not necessarily correspond to one Unicode character!
>>> print(len(test_string))
13
>>> print(len utf8-encoded))
23
>>> print utf8-encodeddecode("utf-8"))
hello! 你怎么是
```

By converting our Unicode codepoints into a sequence of bytes (e.g., via the UTF-8 encoding), weare essentially taking a sequence of codepoints (integers in the range 0 to 154,997) and transforming itinto a sequence of byte values (integers in the range 0 to 255). The 256-length byte vocabulary is muchmore manageable to deal with. When using byte-level tokenization, we do not need to worry about out-of-vocabulary tokens, since we know that any input text can be expressed as a sequence of integers from 0 to255.

# Problem (unicode2): Unicode Encodings (3 points)

(a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather thanUTF-16 or UTF-32? It may be helpful to compare the output of these encodings for variousinput strings.

Deliverable: A one-to-two sentence response.

(b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string intoa Unicode string. Why is this function incorrect? Provide an example of an input byte stringthat yields incorrect results.

```python
defdecodeutf8_bytes_to_str WRONG(bytes): return".join([bytes([b]).decode("utf-8")for b in bytestring])   
>>>decodeutf8_bytes_to_str WRONG("hello".encode("utf-8")) 'hello'
```

Deliverable: An example input byte string for which decode_utf8_bytes_to_str_wrong pro-duces incorrect output, with a one-sentence explanation of why the function is incorrect.

(c) Give a two byte sequence that does not decode to any Unicode character(s).

Deliverable: An example, with a one-sentence explanation.

# 2.3 Subword Tokenization

While byte-level tokenization can alleviate the out-of-vocabulary issues faced by word-level tokenizers, tok-enizing text into bytes results in extremely long input sequences. This slows down model training, since a

sentence with 10 words might only be 10 tokens long in a word-level language model, but could be 50 ormore tokens long in a character-level model (depending on the length of the words). Processing these longersequences requires more computation at each step of the model. Furthermore, language modeling on bytesequences is difficult because the longer input sequences create long-term dependencies in the data.

Subword tokenization is a midpoint between word-level tokenizers and byte-level tokenizers. Note that abyte-level tokenizer’s vocabulary has 256 entries (byte values are 0 to 225). A subword tokenizer trades-off alarger vocabulary size for better compression of the input byte sequence. For example, if the byte sequenceb'the' often occurs in our raw text training data, assigning it an entry in the vocabulary would reduce this3-token sequence to a single token.

How do we select these subword units to add to our vocabulary? Sennrich et al. [2016] propose to usebyte-pair encoding (BPE; Gage, 1994), a compression algorithm that iteratively replaces (“merges”) themost frequent pair of bytes with a single, new unused index. Note that this algorithm adds subword tokensto our vocabulary to maximize the compression of our input sequences—if a word occurs in our input textenough times, it’ll be represented as a single subword unit.

Subword tokenizers with vocabularies constructed via BPE are often called BPE tokenizers. In thisassignment, we’ll implement a byte-level BPE tokenizer, where the vocabulary items are bytes or mergedsequences of bytes, which give us the best of both worlds in terms of out-of-vocabulary handling and man-ageable input sequence lengths. The process of constructing the BPE tokenizer vocabulary is known as“training” the BPE tokenizer.

# 2.4 BPE Tokenizer Training

The BPE tokenizer training procedure consists of three main steps.

Vocabulary initialization The tokenizer vocabulary is a one-to-one mapping from bytestring token tointeger ID. Since we’re training a byte-level BPE tokenizer, our initial vocabulary is simply the set of allbytes. Since there are 256 possible byte values, our initial vocabulary is of size 256.

Pre-tokenization Once you have a vocabulary, you could, in principle, count how often bytes occur nextto each other in your text and begin merging them starting with the most frequent pair of bytes. However,this is quite computationally expensive, since we’d have to go take a full pass over the corpus each timewe merge. In addition, directly merging bytes across the corpus may result in tokens that differ only inpunctuation (e.g., dog! vs. dog.). These tokens would get completely different token IDs, even though theyare likely to have high semantic similarity (since they differ only in punctuation).

To avoid this, we pre-tokenize the corpus. You can think of this as a coarse-grained tokenization over thecorpus that helps us count how often pairs of characters appear. For example, the word 'text' might bea pre-token that appears 10 times. In this case, when we count how often the characters ‘t’ and ‘e’ appearnext to each other, we will see that the word ‘text’ has ‘t’ and ‘e’ adjacent and we can increment their countby 10 instead of looking through the corpus. Since we’re training a byte-level BPE model, each pre-token isrepresented as a sequence of UTF-8 bytes.

The original BPE implementation of Sennrich et al. [2016] pre-tokenizes by simply splitting on whitespace(i.e., s.split(" ")). In contrast, we’ll use a regex-based pre-tokenizer (used by GPT-2; Radford et al., 2019)from github.com/openai/tiktoken/pull/234/files:

>>>PAT  $\equiv$  r""""?:[sdmt]ll|ve|re)！p{L}+|？p{N}+|？[^s\\p{L}\\p{N}]  $+\mid \backslash s + (\text{？！}\backslash S)\mid \backslash s + "''"$

It may be useful to interactively split some text with this pre-tokenizer to get a better sense of itsbehavior:

>>> # requires `regex` package

>>> import regex as re

>>> re.findall(PAT, "some text that i'll pre-tokenize")

['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']

When using it in your code, however, you should use re.finditer to avoid storing the pre-tokenized wordsas you construct your mapping from pre-tokens to their counts.

Compute BPE merges Now that we’ve converted our input text into pre-tokens and represented eachpre-token as a sequence of UTF-8 bytes, we can compute the BPE merges (i.e., train the BPE tokenizer).At a high level, the BPE algorithm iteratively counts every pair of bytes and identifies the pair with thehighest frequency (“A”, “B”). Every occurrence of this most frequent pair (“A”, “B”) is then merged, i.e.,replaced with a new token “AB”. This new merged token is added to our vocabulary; as a result, the finalvocabulary after BPE training is the size of the initial vocabulary (256 in our case), plus the number of BPEmerge operations performed during training. For efficiency during BPE training, we do not consider pairsthat cross pre-token boundaries.2 When computing merges, deterministically break ties in pair frequency bypreferring the lexicographically greater pair. For example, if the pairs (“A”, “B”), (“A”, “C”), (“B”, “ZZ”),and (“BA”, “A”) all have the highest frequency, we’d merge (“BA”, “A”):

```html
>>> max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")]('BA', 'A')
```

Special tokens Often, some strings (e.g., <|endoftext|>) are used to encode metadata (e.g., boundariesbetween documents). When encoding text, it’s often desirable to treat some strings as “special tokens” thatshould never be split into multiple tokens (i.e., will always be preserved as a single token). For example,the end-of-sequence string <|endoftext|> should always be preserved as a single token (i.e., a single integerID), so we know when to stop generating from the language model. These special tokens must be added tothe vocabulary, so they have a corresponding fixed token ID.

Algorithm 1 of Sennrich et al. [2016] contains an inefficient implementation of BPE tokenizer training(essentially following the steps that we outlined above). As a first exercise, it may be useful to implementand test this function to test your understanding.

# Example (bpe_example): BPE training example

Here is a stylized example from Sennrich et al. [2016]. Consider a corpus consisting of the following text

low low low low lowlower lower widest widest widestnewest newest newest newest newest newest

and the vocabulary has a special token <|endoftext|>.

Vocabulary We initialize our vocabulary with our special token <|endoftext|> and the 256 bytevalues.

Pre-tokenization For simplicity and to focus on the merge procedure, we assume in this examplethat pretokenization simply splits on whitespace. When we pretokenize and count, we end up with thefrequency table.

{low: 5, lower: 2, widest: 3, newest: 6}

It is convenient to represent this as a dict[tuple[bytes], int], e.g. $\left\{ { ( 1 , \circ , \mathsf { w } ) } \colon 5 \ldots \right\}$ . Note that evena single byte is a bytes object in Python. There is no byte type in Python to represent a single byte,just as there is no char type in Python to represent a single character.

Merges We first look at every successive pair of bytes and sum the frequency of the words where theyappear {lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 9, st: 9, ne: 6, ew: 6}. The pair $\left( " \thinspace \thinspace \thinspace \thinspace \thinspace \thinspace \right)$and $( \because t ^ { \prime } )$ are tied, so we take the lexicographically greater pair, ('st'). We would then merge thepre-tokens so that we end up with $\big \{ ( 1 , \circ , \mathrm { w } ) \colon 5 $ , (l,o,w,e,r): 2, (w,i,d,e,st): 3, $( \mathtt { n } , \mathtt { e } , \mathtt { w } , \mathtt { e } , \mathtt { s t } ) \colon 6 \}$ .

In the second round, we see that (e, st) is the most common pair (with a count of 9) and we wouldmerge into {(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,est): 3, (n,e,w,est): 6}. Continuing this, thesequence of merges we get in the end will be ['s t', 'e st', 'o w', 'l ow', 'w est', 'n e','ne west', 'w i', 'wi d', 'wid est', 'low e', 'lowe r'].

If we take 6 merges, we have ['s t', 'e st', 'o w', 'l ow', 'w est', 'n e'] and our vocab-ulary elements would be [<|endoftext|>, [...256 BYTE CHARS], st, est, ow, low, west, ne].

With this vocabulary and set of merges, the word newest would tokenize as [ne, west].

# 2.5 Experimenting with BPE Tokenizer Training

Let’s train a byte-level BPE tokenizer on the TinyStories dataset. Instructions to find / download the datasetcan be found in Section 1. Before you start, we recommend taking a look at the TinyStories dataset to geta sense of what’s in the data.

Parallelizing pre-tokenization You will find that a major bottleneck is the pre-tokenization step. Youcan speed up pre-tokenization by parallelizing your code with the built-in library multiprocessing. Con-cretely, we recommend that in parallel implementations of pre-tokenization, you chunk the corpus whileensuring your chunk boundaries occur at the beginning of a special token. You are free to use the startercode at the following link verbatim to obtain chunk boundaries, which you can then use to distribute workacross your processes:

https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/pretokenization_example.py

This chunking will always be valid, since we never want to merge across document boundaries. For thepurposes of the assignment, you can always split in this way. Don’t worry about the edge case of receivinga very large corpus that does not contain <|endoftext|>.

Removing special tokens before pre-tokenization Before running pre-tokenization with the regexpattern (using re.finditer), you should strip out all special tokens from your corpus (or your chunk, if usinga parallel implementation). Make sure that you split on your special tokens, so that no merging can occuracross the text they delimit. For example, if you have a corpus (or chunk) like [Doc 1]<|endoftext|>[Doc2], you should split on the special token <|endoftext|>, and pre-tokenize [Doc 1] and [Doc 2] separately,so that no merging can occur across the document boundary. This can be done using re.split with "|".join(special_tokens) as the delimiter (with careful use of re.escape since | may occur in the specialtokens). The test test_train_bpe_special_tokens will test for this.

Optimizing the merging step The naïve implementation of BPE training in the stylized example aboveis slow because for every merge, it iterates over all byte pairs to identify the most frequent pair. However,the only pair counts that change after each merge are those that overlap with the merged pair. Thus,BPE training speed can be improved by indexing the counts of all pairs and incrementally updating thesecounts, rather than explicitly iterating over each pair of bytes to count pair frequencies. You can getsignificant speedups with this caching procedure, though we note that the merging part of BPE training isnot parallelizable in Python.

# Low-Resource/Downscaling Tip: Profiling

You should use profiling tools like cProfile or scalene to identify the bottlenecks in your imple-mentation, and focus on optimizing those.

# Low-Resource/Downscaling Tip: “Downscaling”

Instead of jumping to training your tokenizer on the full TinyStories dataset, we recommend youfirst train on a small subset of the data: a “debug dataset”. For example, you could train your tokenizeron the TinyStories validation set instead, which is 22K documents instead of 2.12M. This illustrates ageneral strategy of downscaling whenever possible to speed up development: for example, using smallerdatasets, smaller model sizes, etc. Choosing the size of the debug dataset or hyperparameter configrequires careful consideration: you want your debug set to be large enough to have the same bottlenecksas the full configuration (so that the optimizations you make will generalize), but not so big that ittakes forever to run.

# Problem (train_bpe): BPE Tokenizer Training (15 points)

Deliverable: Write a function that, given a path to an input text file, trains a (byte-level) BPEtokenizer. Your BPE training function should handle (at least) the following input parameters:

input_path: str Path to a text file with BPE tokenizer training data.

vocab_size: int A positive integer that defines the maximum final vocabulary size (including theinitial byte vocabulary, vocabulary items produced from merging, and any special tokens).

special_tokens: list[str] A list of strings to add to the vocabulary. These special tokens do nototherwise affect BPE training.

Your BPE training function should return the resulting vocabulary and merges:

vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabu-lary) to bytes (token bytes).

merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list itemis a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with<token2>. The merges should be ordered by order of creation.

To test your BPE training function against our provided tests, you will first need to implement thetest adapter at [adapters.run_train_bpe]. Then, run uv run pytest tests/test_train_bpe.py.Your implementation should be able to pass all tests. Optionally (this could be a large time-investment),you can implement the key parts of your training method using some systems language, for instanceC++ (consider cppyy for this) or Rust (using PyO3). If you do this, be aware of which operationsrequire copying vs reading directly from Python memory, and make sure to leave build instructions, ormake sure it builds using only pyproject.toml. Also note that the GPT-2 regex is not well-supportedin most regex engines and will be too slow in most that do. We have verified that Oniguruma isreasonably fast and supports negative lookahead, but the regex package in Python is, if anything,even faster.

# Problem (train_bpe_tinystories): BPE Training on TinyStories (2 points)

(a) Train a byte-level BPE tokenizer on the TinyStories dataset, using a maximum vocabulary sizeof 10,000. Make sure to add the TinyStories <|endoftext|> special token to the vocabulary.Serialize the resulting vocabulary and merges to disk for further inspection. How many hoursand memory did training take? What is the longest token in the vocabulary? Does it make sense?

Resource requirements: $\leq 3 0$ minutes (no GPUs), ≤ 30GB RAM

Hint You should be able to get under 2 minutes for BPE training using multiprocessing duringpretokenization and the following two facts:

(a) The <|endoftext|> token delimits documents in the data files.

(b) The <|endoftext|> token is handled as a special case before the BPE merges are applied.

Deliverable: A one-to-two sentence response.

(b) Profile your code. What part of the tokenizer training process takes the most time?

Deliverable: A one-to-two sentence response.

Next, we’ll try training a byte-level BPE tokenizer on the OpenWebText dataset. As before, we recom-mend taking a look at the dataset to better understand its contents.

# Problem (train_bpe_expts_owt): BPE Training on OpenWebText (2 points)

(a) Train a byte-level BPE tokenizer on the OpenWebText dataset, using a maximum vocabularysize of 32,000. Serialize the resulting vocabulary and merges to disk for further inspection. Whatis the longest token in the vocabulary? Does it make sense?

Resource requirements: $\leq 1 2$ hours (no GPUs), $\leq$ 100GB RAM

Deliverable: A one-to-two sentence response.

(b) Compare and contrast the tokenizer that you get training on TinyStories versus OpenWebText.

Deliverable: A one-to-two sentence response.

# 2.6 BPE Tokenizer: Encoding and Decoding

In the previous part of the assignment, we implemented a function to train a BPE tokenizer on input textto obtain a tokenizer vocabulary and a list of BPE merges. Now, we will implement a BPE tokenizer thatloads a provided vocabulary and list of merges and uses them to encode and decode text to/from token IDs.

# 2.6.1 Encoding text

The process of encoding text by BPE mirrors how we train the BPE vocabulary. There are a few majorsteps.

Step 1: Pre-tokenize. We first pre-tokenize the sequence and represent each pre-token as a sequence ofUTF-8 bytes, just as we did in BPE training. We will be merging these bytes within each pre-token intovocabulary elements, handling each pre-token independently (no merges across pre-token boundaries).

Step 2: Apply the merges. We then take the sequence of vocabulary element merges created during BPEtraining, and apply it to our pre-tokens in the same order of creation.

# Example (bpe_encoding): BPE encoding example

For example, suppose our input string is 'the cat ate', our vocabulary is {0: b' ', 1: b'a', 2:b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b'at'}, and our learned merges are [(b't', b'h'), (b' ', b'c'), (b' ', 'a'), (b'th', b'e'),(b' a', b't')]. First, our pre-tokenizer would split this string into ['the', ' cat', ' ate'].Then, we’ll look at each pre-token and apply the BPE merges.

The first pre-token 'the' is initially represented as [b't', b'h', b'e']. Looking at our list ofmerges, we identify the first applicable merge to be (b't', b'h'), and use that to transform thepre-token into [b'th', b'e']. Then, we go back to the list of merges and identify the next applicablemerge to be $( { \mathsf { b } } ^ { \prime } \mathsf { t h } ^ { \prime } , \mathsf { b } ^ { \prime } { \mathsf { e } } ^ { \prime } )$ , which transforms the pre-token into [b'the']. Finally, looking back atthe list of merges, we see that there are no more that apply to the string (since the entire pre-tokenhas been merged into a single token), so we are done applying the BPE merges. The correspondinginteger sequence is [9].

Repeating this process for the remaining pre-tokens, we see that the pre-token ' cat' is representedas [b' c', b'a', b't'] after applying the BPE merges, which becomes the integer sequence [7, 1,5]. The final pre-token ' ate' is [b' at', b'e'] after applying the BPE merges, which becomes theinteger sequence [10, 3]. Thus, the final result of encoding our input string is [9, 7, 1, 5, 10,3].

Special tokens. Your tokenizer should be able to properly handle user-defined special tokens when encod-ing text (provided when constructing the tokenizer).

Memory considerations. Suppose we want to tokenize a large text file that we cannot fit in memory.To efficiently tokenize this large file (or any other stream of data), we need to break it up into manageablechunks and process each chunk in-turn, so that the memory complexity is constant as opposed to linear inthe size of the text. In doing so, we need to make sure that a token doesn’t cross chunk boundaries, elsewe’ll get a different tokenization than the naïve method of tokenizing the entire sequence in-memory.

# 2.6.2 Decoding text

To decode a sequence of integer token IDs back to raw text, we can simply look up each ID’s correspondingentries in the vocabulary (a byte sequence), concatenate them together, and then decode the bytes to aUnicode string. Note that input IDs are not guaranteed to map to valid Unicode strings (since a usercould input any sequence of integer IDs). In the case that the input token IDs do not produce a validUnicode string, you should replace the malformed bytes with the official Unicode replacement characterU+FFFD.3 The errors argument of bytes.decode controls how Unicode decoding errors are handled, andusing errors $=$ 'replace' will automatically replace malformed data with the replacement marker.

# Problem (tokenizer): Implementing the tokenizer (15 points)

Deliverable: Implement a Tokenizer class that, given a vocabulary and a list of merges, encodestext into integer IDs and decodes integer IDs into text. Your tokenizer should also support user-providedspecial tokens (appending them to the vocabulary if they aren’t already there). We recommend thefollowing interface:

def __init__(self, vocab, merges, special_tokens=None) Construct a tokenizer from a givenvocabulary, list of merges, and (optionally) a list of special tokens. This function should accept

```python
the following parameters:  
vocab: dict[int, bytes]  
merges: list[tuple[bytes, bytes]]  
special_tokens: list[str] | None = None
```

```python
def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None) Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges (in the same format that your BPE training code output) and (optionally) a list of special tokens. This method should accept the following additional parameters:
```

```txt
vocab_filepath: str  
merges_filepath: str  
special_tokens: list[str] | None = None
```

```txt
def encode(self, text: str) -> list[int] Encode an input text into a sequence of token IDs.
```

```txt
def encode iterable(self, iterable: Iterator[str]) -> Iterator[int] Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
```

```txt
defdecode(self，ids：list[int]）->strDecodeasequenceoftokenIDsinto text.
```

```txt
To test your Tokenizer against our provided tests, you will first need to implement the test adapter at [adapters.get_tokenizer]. Then, run uv run pytest tests/test_tokenizer.py. Your implementation should be able to pass all tests.
```

# 2.7 Experiments


Problem (tokenizer_experiments): Experiments with tokenizers (4 points)


(a) Sample 10 documents from TinyStories and OpenWebText. Using your previously-trained TinyS-tories and OpenWebText tokenizers (10K and 32K vocabulary size, respectively), encode thesesampled documents into integer IDs. What is each tokenizer’s compression ratio (bytes/token)?

Deliverable: A one-to-two sentence response.

(b) What happens if you tokenize your OpenWebText sample with the TinyStories tokenizer? Com-pare the compression ratio and/or qualitatively describe what happens.

Deliverable: A one-to-two sentence response.

(c) Estimate the throughput of your tokenizer (e.g., in bytes/second). How long would it take totokenize the Pile dataset (825GB of text)?

Deliverable: A one-to-two sentence response.

(d) Using your TinyStories and OpenWebText tokenizers, encode the respective training and devel-opment datasets into a sequence of integer token IDs. We’ll use this later to train our languagemodel. We recommend serializing the token IDs as a NumPy array of datatype uint16. Why isuint16 an appropriate choice?

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-19/c306355c-da21-4a16-9f4a-0d7eb8fed997/a97ad0f1cf6a74d92b385dda5866aa2d065e069af00d828a3c4383bb1122cc78.jpg)



Figure 1: An overview of our Transformer languagemodel.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-19/c306355c-da21-4a16-9f4a-0d7eb8fed997/9fada81f8310e9cbc8ffc6650bd334d5f303945ff3720b2bde57ff03beb51e06.jpg)



Figure 2: A pre-norm Transformer block.


# 3 Transformer Language Model Architecture

A language model takes as input a batched sequence of integer token IDs (i.e., torch.Tensor of shape(batch_size, sequence_length)), and returns a (batched) normalized probability distribution over thevocabulary (i.e., a PyTorch Tensor of shape (batch_size, sequence_length, vocab_size)), where thepredicted distribution is over the next word for each input token. When training the language model, weuse these next-word predictions to calculate the cross-entropy loss between the actual next word and thepredicted next word. When generating text from the language model during inference, we take the predictednext-word distribution from the final time step (i.e., the last item in the sequence) to generate the next tokenin the sequence (e.g., by taking the token with the highest probability, sampling from the distribution, etc.),add the generated token to the input sequence, and repeat.

In this part of the assignment, you will build this Transformer language model from scratch. We willbegin with a high-level description of the model before progressively detailing the individual components.

# 3.1 Transformer LM

Given a sequence of token IDs, the Transformer language model uses an input embedding to convert tokenIDs to dense vectors, passes the embedded tokens through num_layers Transformer blocks, and then appliesa learned linear projection (the “output embedding” or “LM head”) to produce the predicted next-tokenlogits. See Figure 1 for a schematic representation.

# 3.1.1 Token Embeddings

In the very first step, the Transformer embeds the (batched) sequence of token IDs into a sequence of vectorscontaining information on the token identity (red blocks in Figure 1).

More specifically, given a sequence of token IDs, the Transformer language model uses a token em-bedding layer to produce a sequence of vectors. Each embedding layer takes in a tensor of integersof shape (batch_size, sequence_length) and produces a sequence of vectors of shape (batch_size,sequence_length, d_model).

# 3.1.2 Pre-norm Transformer Block

After embedding, the activations are processed by several identically structured neural net layers. A standarddecoder-only Transformer language model consists of num_layers identical layers (commonly called Trans-former “blocks”). Each Transformer block takes in an input of shape (batch_size, sequence_length,d_model) and returns an output of shape (batch_size, sequence_length, d_model). Each block aggre-gates information across the sequence (via self-attention) and non-linearly transforms it (via the feed-forwardlayers).

# 3.2 Output Normalization and Embedding

After num_layers Transformer blocks, we will take the final activations and turn them into a distributionover the vocabulary.

We will implement the “pre-norm” Transformer block (detailed in §3.5), which additionally requires theuse of layer normalization (detailed below) after the final Transformer block to ensure its outputs are properlyscaled.

After this normalization, we will use a standard learned linear transformation to convert the output ofthe Transformer blocks into predicted next-token logits (see, e.g., Radford et al. [2018] equation 2).

# 3.3 Remark: Batching, Einsum and Efficient Computation

Throughout the Transformer, we will be performing the same computation applied to many batch-like inputs.Here are a few examples:

• Elements of a batch: we apply the same Transformer forward operation on each batch element.

• Sequence length: the “position-wise” operations like RMSNorm and feed-forward operate identicallyon each position of a sequence.

• Attention heads: the attention operation is batched across attention heads in a “multi-headed”attention operation.

It is useful to have an ergonomic way of performing such operations in a way that fully utilizes the GPU,and is easy to read and understand. Many PyTorch operations can take in excess “batch-like” dimensionsat the start of a tensor and repeat/broadcast the operation across these dimensions efficiently.

For instance, say we are doing a position-wise, batched operation. We have a “data tensor” $D$ of shape(batch_size, sequence_length, d_model), and we would like to do a batched vector-matrix multiplyagainst a matrix $A$ of shape (d_model, d_model). In this case, D @ A will do a batched matrix multiply,which is an efficient primitive in PyTorch, where the (batch_size, sequence_length) dimensions arebatched over.

Because of this, it is helpful to assume that your functions may be given additional batch-like dimensionsand to keep those dimensions at the start of the PyTorch shape. To organize tensors so they can be batchedin this manner, they might need to be shaped using many steps of view, reshape and transpose. This canbe a bit of a pain, and it often gets hard to read what the code is doing and what the shapes of your tensorsare.

A more ergonomic option is to use einsum notation within torch.einsum, or rather use frameworkagnostic libraries like einops or einx. The two key ops are einsum, which can do tensor contractions witharbitrary dimensions of input tensors, and rearrange, which can reorder, concatenate, and split arbitrary

dimensions. It turns out almost all operations in machine learning are some combination of dimensionjuggling and tensor contraction with the occasional (usually pointwise) nonlinear function. This means thata lot of your code can be more readable and flexible when using einsum notation.

We strongly recommend learning and using einsum notation for the class. Students who have notbeen exposed to einsum notation before should use einops (docs here), and students who are alreadycomfortable with einops should learn the more general einx (here).4 Both packages are already installedin the environment we’ve supplied.

Here we give some examples of how einsum notation can be used. These are a supplement to thedocumentation for einops, which you should read first.


Example (einstein_example1): Batched matrix multiplication with einops.einsum


```python
import torch   
from einops import rearrange, einsum   
## Basic implementation   
Y = D @ A.T   
# Hard to tell the input and output shapes and what they mean.   
# What shapes can D and A have, and do any of these have unexpected behavior?   
## Einsum is self-documenting and robust   
# D A -> Y   
Y = einsum(D,A,"batch sequence d_in, d_out d_in -> batch sequence d_out")   
## Or, a batched version where D can have any leading dimensions but A is constrained.   
Y = einsum(D,A,"... d_in, d_out d_in -> ... d_out")
```


Example (einstein_example2): Broadcasted operations with einops.rearrange


We have a batch of images, and for each image we want to generate 10 dimmed versions based on some scaling factor:   
images  $=$  torch RANDn(64, 128, 128, 3) # (batch, height, width, channel)   
dim_by  $=$  torch.linspace(start=0.0, end=1.0, steps=10)   
Reshape and multiply   
dim_value  $=$  rearrange(dim_by, "dim_value -> 1 dim_value 1 1 1")   
images_rearr  $=$  rearrange/images, "b height width channel -> b 1 height width channel")   
dimmed_images  $=$  images_rearr \* dim_value   
# Or in one go:   
dimmed_images  $=$  einsum( images, dim_by, "batch height width channel, dim_value -> batch dim_value height width channel"


Example (einstein_example3): Pixel mixing with einops.rearrange


Suppose we have a batch of images represented as a tensor of shape (batch, height, width, channel), and we want to perform a linear transformation across all pixels of the image, but this transformation should happen independently for each channel. Our linear transformation is represented as a matrix  $B$  of shape (height  $\times$  width, height  $\times$  width). channels_last = torch randn(64, 32, 32, 3) # (batch, height, width, channel) B = torch randn(32*32, 32*32)   
#Rearrange an image tensor for mixing across all pixels channels_last-flat = channels_last.view( -1, channels_last.size(1) * channels_last.size(2), channels_last.size(3)   
) channels_firstflat  $\equiv$  channels_lastflat.transpose(1, 2)   
channels_firstflattransformed  $\equiv$  channels_firstflat @ B.T   
channels_lastflattransformed  $\equiv$  channels_firstflattransformed.transpose(1, 2)   
channels_lasttransformed  $\equiv$  channels_lastflattransformed.view(*channels_last.shape)   
Instead, using einops:   
height  $=$  width  $= 32$    
#Rearrange replaces clunky torch view + transpose   
channels_first  $\equiv$  rearrange( channels_last, "batch height width channel -> batch channel (height width)"   
)   
channels_firsttransformed  $\equiv$  einsum( channels_first, B, "batch channel pixel_in, pixel_out pixel_in -> batch channel pixel_out"   
)   
channels_lasttransformed  $\equiv$  rearrange( channels_first_transformed, "batch channel (height width) -> batch height width channel", height  $\equiv$  height, width  $\equiv$  width   
)   
Or, if you're feeling crazy: all in one go using einx.dot (einx equivalent of einops.einsum)   
height  $=$  width  $= 32$    
channels_lasttransformed  $\equiv$  einx.dot( "batch row_in col_in channel, (row_out col_out) (row_in col_in)" "-> batch row_out col_out channel", channels_last, B, col_in  $\equiv$  width, col_out  $\equiv$  width

The first implementation here could be improved by placing comments before and after to indicate

what the input and output shapes are, but this is clunky and susceptible to bugs. With einsumnotation, documentation is implementation!

Einsum notation can handle arbitrary input batching dimensions, but also has the key benefit of beingself-documenting. It’s much clearer what the relevant shapes of your input and output tensors are in codethat uses einsum notation. For the remaining tensors, you can consider using Tensor type hints, for instanceusing the jaxtyping library (not specific to Jax).

We will talk more about the performance implications of using einsum notation in assignment 2, but fornow know that they’re almost always better than the alternative!

# 3.3.1 Mathematical Notation and Memory Ordering

Many machine learning papers use row vectors in their notation, which result in representations that meshwell with the row-major memory ordering used by default in NumPy and PyTorch. With row vectors, alinear transformation looks like

$$
y = x W ^ {\top}, \tag {1}
$$

for row-major $W \in \mathbb { R } ^ { d _ { \mathrm { o u t } } \times d _ { \mathrm { i n } } }$ and row-vector $\boldsymbol { x } \in \mathbb { R } ^ { 1 \times d _ { \mathrm { i n } } }$ .

In linear algebra it’s generally more common to use column vectors, where linear transformations looklike

$$
y = W x, \tag {2}
$$

given a row-major $W \in \mathbb { R } ^ { d _ { \mathrm { o u t } } \times d _ { \mathrm { i n } } }$ and column-vector $x \in \mathbb { R } ^ { d _ { \mathrm { i n } } }$ . We will use column vectors for mathe-matical notation in this assignment, as it is generally easier to follow the math this way. You should keep inmind that if you want to use plain matrix multiplication notation, you will have to apply matrices using therow vector convention, since PyTorch uses row-major memory ordering. If you use einsum for your matrixoperations, this should be a non-issue.

# 3.4 Basic Building Blocks: Linear and Embedding Modules

# 3.4.1 Parameter Initialization

Training neural networks effectively often requires careful initialization of the model parameters—bad initial-izations can lead to undesirable behavior such as vanishing or exploding gradients. Pre-norm transformersare unusually robust to initializations, but they can still have a siginificant impact on training speed andconvergence. Since this assignment is already long, we will save the details for assignment 3, and insteadgive you some approximate initializations that should work well for most cases. For now, use:

• Linear weights: $\begin{array} { r } { \mathcal { N } \left( \mu = 0 , \sigma ^ { 2 } = \frac { 2 } { d _ { \mathrm { i n } } + d _ { \mathrm { o u t } } } \right) } \end{array}$ truncated at $[ - 3 \sigma , 3 \sigma ]$

• Embedding: $\mathcal { N } \left( \mu = 0 , \sigma ^ { 2 } = 1 \right)$ truncated at $[ - 3 , 3 ]$

• RMSNorm: 1

You should use torch.nn.init.trunc_normal_ to initialize the truncated normal weights.

# 3.4.2 Linear Module

Linear layers are a fundamental building block of Transformers and neural nets in general. First, you willimplement your own Linear class that inherits from torch.nn.Module and performs a linear transformation:

$$
y = W x. \tag {3}
$$

Note that we do not include a bias term, following most modern LLMs.

# Problem (linear): Implementing the linear module (1 point)

Deliverable: Implement a Linear class that inherits from torch.nn.Module and performs a lineartransformation. Your implementation should follow the interface of PyTorch’s built-in nn.Linearmodule, except for not having a bias argument or parameter. We recommend the following interface:

def __init__(self, in_features, out_features, device=None, dtype=None) Construct alinear transformation module. This function should accept the following parameters:

in_features: int final dimension of the input

out_features: int final dimension of the output

device: torch.device | None $=$ None Device to store the parameters on

dtype: torch.dtype | None $=$ None Data type of the parameters

def forward(self, x: torch.Tensor) $- >$ torch.Tensor Apply the linear transformation to theinput.

Make sure to:

• subclass nn.Module

• call the superclass constructor

• construct and store your parameter as $W$ (not $W ^ { \parallel }$ ) for memory ordering reasons, putting it inan nn.Parameter

• of course, don’t use nn.Linear or nn.functional.linear

For initializations, use the settings from above along with torch.nn.init.trunc_normal_ toinitialize the weights.

To test your Linear module, implement the test adapter at [adapters.run_linear]. The adaptershould load the given weights into your Linear module. You can use Module.load_state_dict forthis purpose. Then, run uv run pytest -k test_linear.

# 3.4.3 Embedding Module

As discussed above, the first layer of the Transformer is an embedding layer that maps integer token IDsinto a vector space of dimension d_model. We will implement a custom Embedding class that inherits fromtorch.nn.Module (so you should not use nn.Embedding). The forward method should select the embeddingvector for each token ID by indexing into an embedding matrix of shape (vocab_size, d_model) using atorch.LongTensor of token IDs with shape (batch_size, sequence_length).

# Problem (embedding): Implement the embedding module (1 point)

Deliverable: Implement the Embedding class that inherits from torch.nn.Module and performs anembedding lookup. Your implementation should follow the interface of PyTorch’s built-innn.Embedding module. We recommend the following interface:

def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None) Constructan embedding module. This function should accept the following parameters:

num_embeddings: int Size of the vocabulary

embedding_dim: int Dimension of the embedding vectors, i.e., $d _ { \mathrm { m o d e l } }$

device: torch.device | None $=$ None Device to store the parameters on

dtype: torch.dtype | None $=$ None Data type of the parameters

def forward(self, token_ids: torch.Tensor) $- >$ torch.Tensor Lookup the embedding vectorsfor the given token IDs.

Make sure to:

• subclass nn.Module

• call the superclass constructor

• initialize your embedding matrix as a nn.Parameter

• store the embedding matrix with the d_model being the final dimension

• of course, don’t use nn.Embedding or nn.functional.embedding

Again, use the settings from above for initialization, and use torch.nn.init.trunc_normal_ toinitialize the weights.

To test your implementation, implement the test adapter at [adapters.run_embedding]. Then, runuv run pytest -k test_embedding.

# 3.5 Pre-Norm Transformer Block

Each Transformer block has two sub-layers: a multi-head self-attention mechanism and a position-wisefeed-forward network (Vaswani et al., 2017, section 3.1).

In the original Transformer paper, the model uses a residual connection around each of the two sub-layers,followed by layer normalization. This architecture is commonly known as the “post-norm” Transformer, sincelayer normalization is applied to the sublayer output. However, a variety of work has found that movinglayer normalization from the output of each sub-layer to the input of each sub-layer (with an additionallayer normalization after the final Transformer block) improves Transformer training stability [Nguyen andSalazar, 2019, Xiong et al., 2020]—see Figure 2 for a visual representation of this “pre-norm” Transformerblock. The output of each Transformer block sub-layer is then added to the sub-layer input via the residualconnection (Vaswani et al., 2017, section 5.4). An intuition for pre-norm is that there is a clean “residualstream” without any normalization going from the input embeddings to the final output of the Transformer,which is purported to improve gradient flow. This pre-norm Transformer is now the standard used in languagemodels today (e.g., GPT-3, LLaMA, PaLM, etc.), so we will implement this variant. We will walk througheach of the components of a pre-norm Transformer block, implementing them in sequence.

# 3.5.1 Root Mean Square Layer Normalization

The original Transformer implementation of Vaswani et al. [2017] uses layer normalization [Ba et al., 2016]to normalize activations. Following Touvron et al. [2023], we will use root mean square layer normalization(RMSNorm; Zhang and Sennrich, 2019, equation 4) for layer normalization. Given a vector $a \in \mathbb { R } ^ { d _ { \mathrm { m o d e l } } }$ ofactivations, RMSNorm will rescale each activation $a _ { i }$ as follows:

$$
\operatorname {R M S N o r m} \left(a _ {i}\right) = \frac {a _ {i}}{\operatorname {R M S} (a)} g _ {i}, \tag {4}
$$

where RMS(a) = $\begin{array} { r } { \mathrm { R M S } ( a ) = \sqrt { \frac { 1 } { d _ { \mathrm { m o d e l } } } \sum _ { i = 1 } ^ { d _ { \mathrm { m o d e l } } } a _ { i } ^ { 2 } + \varepsilon } } \end{array}$ dmodel . Here, $g _ { i }$ is a learnable “gain” parameter (there are d_model suchparameters total), and $\varepsilon$ is a hyperparameter that is often fixed at 1e-5.

You should upcast your input to torch.float32 to prevent overflow when you square the input. Overall,your forward method should look like:

in dtype  $=$  x.dtype   
 $\mathbf{x} = \mathbf{x}$  to(torch.float32)

# Your code here performing RMSNorm

result  $= \dots$

# Return the result in the original dtype

return result.to(in_dtype)

# Problem (rmsnorm): Root Mean Square Layer Normalization (1 point)

Deliverable: Implement RMSNorm as a torch.nn.Module. We recommend the following interface:

```python
def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None) Construct the RMSNorm module. This function should accept the following parameters:
```

d_model: int Hidden dimension of the model

eps: float = 1e-5 Epsilon value for numerical stability

device: torch.device | None $=$ None Device to store the parameters on

dtype: torch.dtype | None $=$ None Data type of the parameters

def forward(self, x: torch.Tensor) $- >$ torch.Tensor Process an input tensor of shape

(batch_size, sequence_length, d_model) and return a tensor of the same shape.

Note: Remember to upcast your input to torch.float32 before performing the normalization (andlater downcast to the original dtype), as described above.

To test your implementation, implement the test adapter at [adapters.run_rmsnorm]. Then, run uvrun pytest -k test_rmsnorm.

# 3.5.2 Position-Wise Feed-Forward Network

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-19/c306355c-da21-4a16-9f4a-0d7eb8fed997/9f844880fc008b93b1f876c395e829dfe81b498dda5c8a94dc57bc86252dc74c.jpg)



Figure 3: Comparing the SiLU (aka Swish) and ReLU activation functions.


In the original Transformer paper (section 3.3 of Vaswani et al. [2017]), the Transformer feed-forward networkconsists of two linear transformations with a ReLU activation $( \mathrm { R e L U } ( x ) = \operatorname* { m a x } ( 0 , x ) )$ ) between them. Thedimensionality of the inner feed-forward layer is typically 4x the input dimensionality.

However, modern language models tend to incorporate two main changes compared to this original design:they use another activation function and employ a gating mechanism. Specifically, we will implement the“SwiGLU” activation function adopted in LLMs like Llama 3 [Grattafiori et al., 2024] and Qwen 2.5 [Yanget al., 2024], which combines the SiLU (often called Swish) activation with a gating mechanism called aGated Linear Unit (GLU). We will also omit the bias terms sometimes used in linear layers, following mostmodern LLMs since PaLM [Chowdhery et al., 2022] and LLaMA [Touvron et al., 2023].

The SiLU or Swish activation function [Hendrycks and Gimpel, 2016, Elfwing et al., 2017] is defined asfollows:

$$
\operatorname {S i L U} (x) = x \cdot \sigma (x) = \frac {x}{1 + e ^ {- x}} \tag {5}
$$

As can be seen in Figure 3, the SiLU activation function is similar to the ReLU activation function, butis smooth at zero.

Gated Linear Units (GLUs) were originally defined by Dauphin et al. [2017] as the element-wise productof a linear transformation passed through a sigmoid function and another linear transformation:

$$
\operatorname {G L U} (x, W _ {1}, W _ {2}) = \sigma \left(W _ {1} x\right) \odot W _ {2} x, \tag {6}
$$

where $\odot$ represents element-wise multiplication. Gated Linear Units are suggested to “reduce the vanishinggradient problem for deep architectures by providing a linear path for the gradients while retaining non-linearcapabilities.”

Putting the SiLU/Swish and GLU together, we get the SwiGLU, which we will use for our feed-forwardnetworks:

$$
\operatorname {F F N} (x) = \operatorname {S w i G L U} (x, W _ {1}, W _ {2}, W _ {3}) = W _ {2} (\operatorname {S i L U} (W _ {1} x) \odot W _ {3} x), \tag {7}
$$

where $x \in \mathbb { R } ^ { d _ { \mathrm { m o d e l } } }$ , $W _ { 1 } , W _ { 3 } \in \mathbb { R } ^ { d _ { \mathrm { f f } } \times d _ { \mathrm { m o d e l } } }$ , $W _ { 2 } \in \mathbb { R } ^ { d _ { \mathrm { m o d e l } } \times d _ { \mathrm { f f } } }$ , and canonically, $d _ { \mathrm { f f } } = \frac { \mathrm { s } } { 3 } d _ { \mathrm { m o d e l } }$

Shazeer [2020] first proposed combining the SiLU/Swish activation with GLUs and conducted experimentsshowing that SwiGLU outperforms baselines like ReLU and SiLU (without gating) on language modelingtasks. Later in the assignment, you will compare SwiGLU and SiLU. Though we’ve mentioned some heuristicarguments for these components (and the papers provide more supporting evidence), it’s good to keep anempirical perspective: a now famous quote from Shazeer’s paper is

We offer no explanation as to why these architectures seem to work; we attribute their success,as all else, to divine benevolence.

# Problem (positionwise_feedforward): Implement the position-wise feed-forward network(2 points)

Deliverable: Implement the SwiGLU feed-forward network, composed of a SiLU activationfunction and a GLU.

Note: in this particular case, you should feel free to use torch.sigmoid in your implementationfor numerical stability.

You should set $d _ { \mathrm { f f } }$ to approximately ${ \begin{array} { l } { { \frac { 8 } { 3 } } \ \times \ d _ { \mathrm { m o d e l } } } \end{array} }$ in your implementation, while ensuring thatthe dimensionality of the inner feed-forward layer is a multiple of 64 to make good use of yourhardware. To test your implementation against our provided tests, you will need to implementthe test adapter at [adapters.run_swiglu]. Then, run uv run pytest -k test_swiglu totest your implementation.

# 3.5.3 Relative Positional Embeddings

To inject positional information into the model, we will implement Rotary Position Embeddings [Su et al.,2021], often called RoPE. For a given query token $q ^ { ( i ) } = W _ { q } x ^ { ( i ) } \in \mathbb { R } ^ { d }$ at token position $_ i$ , we will apply apairwise rotation matrix $R ^ { i }$ , giving us $q ^ { \prime ( i ) } = R ^ { i } q ^ { ( i ) } = R ^ { i } W _ { q } x ^ { ( i ) }$ . Here, $R ^ { i }$ will rotate pairs of embeddingelements $q _ { 2 k - 1 : 2 k } ^ { ( i ) }$ as 2d vectors by the angle $\begin{array} { r } { \theta _ { i , k } = \frac { i } { \Theta ^ { ( 2 k - 2 ) / d } } } \end{array}$ for $k \in \{ 1 , \ldots , d / 2 \}$ and some constant $\Theta$ . Thus,we can consider $R ^ { i }$ to be a block-diagonal matrix of size $d \times d$ , with blocks $R _ { k } ^ { i }$ for $k \in \{ 1 , \ldots , d / 2 \}$ , with

$$
R _ {k} ^ {i} = \left[ \begin{array}{c c} \cos (\theta_ {i, k}) & - \sin (\theta_ {i, k}) \\ \sin (\theta_ {i, k}) & \cos (\theta_ {i, k}) \end{array} \right]. \tag {8}
$$

Thus we get the full rotation matrix

$$
R ^ {i} = \left[ \begin{array}{c c c c c} R _ {1} ^ {i} & 0 & 0 & \dots & 0 \\ 0 & R _ {2} ^ {i} & 0 & \dots & 0 \\ 0 & 0 & R _ {3} ^ {i} & \dots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \dots & R _ {d / 2} ^ {i} \end{array} \right], \tag {9}
$$

where 0s represent $2 \times 2$ zero matrices. While one could construct the full $d \times d$ matrix, a good solutionshould use the properties of this matrix to implement the transformation more efficiently. Since we onlycare about the relative rotation of tokens within a given sequence, we can reuse the values we compute for$\cos ( \theta _ { i , k } )$ and $\sin ( \theta _ { i , k } )$ across layers, and different batches. If you would like to optimize it, you may use asingle RoPE module referenced by all layers, and it can have a 2d pre-computed buffer of sin and cos valuescreated during init with self.register_buffer(persistent=False), instead of a nn.Parameter (becausewe do not want to learn these fixed cosine and sine values). The exact same rotation process we did forour $q ^ { ( i ) }$ is then done for $k ^ { ( j ) }$ , rotating by the corresponding $R ^ { j }$ . Notice that this layer has no learnableparameters.

# Problem (rope): Implement RoPE (2 points)

Deliverable: Implement a class RotaryPositionalEmbedding that applies RoPE to the inputtensor. The following interface is recommended:

def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) Constructthe RoPE module and create buffers if needed.

theta: float $\Theta$ value for the RoPE

d_k: int dimension of query and key vectors

max_seq_len: int Maximum sequence length that will be inputted

device: torch.device | None $=$ None Device to store the buffer on

def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor

Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.

Note that you should tolerate $x$ with an arbitrary number of batch dimensions. You shouldassume that the token positions are a tensor of shape (..., seq_len) specifying the tokenpositions of $x$ along the sequence dimension.

You should use the token positions to slice your (possibly precomputed) cos and sin tensorsalong the sequence dimension.

To test your implementation, complete [adapters.run_rope] and make sure it passes uv runpytest -k test_rope.

# 3.5.4 Scaled Dot-Product Attention

We will now implement scaled dot-product attention as described in Vaswani et al. [2017] (section 3.2.1).As a preliminary step, the definition of the Attention operation will make use of softmax, an operation thattakes an unnormalized vector of scores and turns it into a normalized distribution:

$$
\operatorname {s o f t m a x} (v) _ {i} = \frac {\exp \left(v _ {i}\right)}{\sum_ {j = 1} ^ {n} \exp \left(v _ {j}\right)}. \tag {10}
$$

Note that $\exp ( v _ { i } )$ can become inf for large values (then, inf/inf = NaN). We can avoid this by noticingthat the softmax operation is invariant to adding any constant $c$ to all inputs. We can leverage this propertyfor numerical stability—typically, we will subtract the largest entry of $o _ { i }$ from all elements of $o _ { i }$ , making thenew largest entry 0. You will now implement softmax, using this trick for numerical stability.

# Problem (softmax): Implement softmax (1 point)

Deliverable: Write a function to apply the softmax operation on a tensor. Your function shouldtake two parameters: a tensor and a dimension $i$ , and apply softmax to the $\imath$ -th dimension of the inputtensor. The output tensor should have the same shape as the input tensor, but its $i$ -th dimension willnow have a normalized probability distribution. Use the trick of subtracting the maximum value inthe $i$ -th dimension from all elements of the $i$ -th dimension to avoid numerical stability issues.

To test your implementation, complete [adapters.run_softmax] and make sure it passes uv runpytest -k test_softmax_matches_pytorch.

We can now define the Attention operation mathematically as follows:

$$
\operatorname {A t t e n t i o n} (Q, K, V) = \operatorname {s o f t m a x} \left(\frac {Q ^ {\top} K}{\sqrt {d _ {k}}}\right) V \tag {11}
$$

where $Q \in \mathbb { R } ^ { n \times d _ { k } }$ , $K \in \mathbb { R } ^ { m \times d _ { k } }$ , and $V \in \mathbb { R } ^ { m \times d _ { v } }$ . Here, $Q$ , $K$ and $V$ are all inputs to this operation—notethat these are not the learnable parameters. If you’re wondering why this isn’t $Q K ^ { \top }$ , see 3.3.1.

Masking: It is sometimes convenient to mask the output of an attention operation. A mask should havethe shape $M \in \{ \mathrm { T r u e } , \mathrm { F a l } \mathbf { s } \mathbf { e } \} ^ { n \times m }$ , and each row $_ i$ of this boolean matrix indicates which keys the query$i$ should attend to. Canonically (and slightly confusingly), a value of True at position $( i , j )$ indicates thatthe query $i$ does attend to the key $j$ , and a value of False indicates that the query does not attend to thekey. In other words, “information flows” at $( i , j )$ pairs with value True. For example, consider a $1 \times 3$ maskmatrix with entries [[True, True, False]]. The single query vector attends only to the first two keys.

Computationally, it will be much more efficient to use masking than to compute attention on subse-quences, and we can do this by taking the pre-softmax values $\left( { \frac { Q ^ { \top } K } { \sqrt { d _ { k } } } } \right)$ and adding a $- \infty$ in any entry of themask matrix that is False.

# Problem (scaled_dot_product_attention): Implement scaled dot-product attention(5 points)

Deliverable: Implement the scaled dot-product attention function. Your implementation shouldhandle keys and queries of shape (batch_size, ..., seq_len, d_k) and values of shape(batch_size, ..., seq_len, d_v), where ... represents any number of other batch-likedimensions (if provided). The implementation should return an output with the shape (batch_size,..., d_v). See section 3.3 for a discussion on batch-like dimensions.

Your implementation should also support an optional user-provided boolean mask of shape (seq_len,seq_len). The attention probabilities of positions with a mask value of True should collectively sumto 1, and the attention probabilities of positions with a mask value of False should be zero.To test your implementation against our provided tests, you will need to implement the test adapterat [adapters.run_scaled_dot_product_attention].

uv run pytest -k test_scaled_dot_product_attention tests your implementation on third-orderinput tensors, while uv run pytest -k test_4d_scaled_dot_product_attention tests yourimplementation on fourth-order input tensors.

# 3.5.5 Causal Multi-Head Self-Attention

We will implement multi-head self-attention as described in section 3.2.2 of Vaswani et al. [2017]. Recallthat, mathematically, the operation of applying multi-head attention is defined as follows:

$$
\operatorname {M u l t i H e a d} (Q, K, V) = \operatorname {C o n c a t} \left(\operatorname {h e a d} _ {1}, \dots , \operatorname {h e a d} _ {h}\right) \tag {12}
$$

$$
\text {f o r} \quad \operatorname {h e a d} _ {i} = \operatorname {A t t e n t i o n} \left(Q _ {i}, K _ {i}, V _ {i}\right) \tag {13}
$$

with $Q _ { i }$ , $K _ { i }$ , $V _ { i }$ being slice number $i \in \{ 1 , \ldots , h \}$ of size $d _ { k }$ or $d _ { v }$ of the embedding dimension for $Q , K$ , and$V$ respectively. With Attention being the scaled dot-product attention operation defined in §3.5.4. Fromthis we can form the multi-head self -attention operation:

$$
\operatorname {M u l t i H e a d S e l f A t t e n t i o n} (x) = W _ {O} \operatorname {M u l t i H e a d} \left(W _ {Q} x, W _ {K} x, W _ {V} x\right) \tag {14}
$$

Here, the learnable parameters are $W _ { Q } \in \mathbb { R } ^ { h d _ { k } \times d _ { \mathrm { m o d e l } } }$ , $W _ { K } \in \mathbb { R } ^ { h d _ { k } \times d _ { \mathrm { m o d e l } } }$ , $W _ { V } \in \mathbb R ^ { h d _ { v } \times d _ { \mathrm { m o d e l } } }$ , and $W _ { O } \in$$\mathbb { R } ^ { d _ { \mathrm { m o d e l } } \times h d _ { v } }$ . Since the Qs, $K$ , and V s are sliced in the multi-head attention operation, we can think of $W _ { Q }$ ,$W _ { K }$ and $W _ { V }$ as being separated for each head along the output dimension. When you have this working,you should be computing the key, value, and query projections in a total of three matrix multiplies.5

Causal masking. Your implementation should prevent the model from attending to future tokens in thesequence. In other words, if the model is given a token sequence $t _ { 1 } , \ldots , t _ { n }$ , and we want to calculate thenext-word predictions for the prefix $t _ { 1 } , \ldots , t _ { i }$ (where $i < n$ ), the model should not be able to access (attendto) the token representations at positions $t _ { i + 1 } , \ldots , t _ { n }$ since it will not have access to these tokens whengenerating text during inference (and these future tokens leak information about the identity of the truenext word, trivializing the language modeling pre-training objective). For an input token sequence $t _ { 1 } , \ldots , t _ { n }$we can naively prevent access to future tokens by running multi-head self-attention $n$ times (for the $n$ uniqueprefixes in the sequence). Instead, we’ll use causal attention masking, which allows token $i$ to attend to allpositions $j \le i$ in the sequence. You can use torch.triu or a broadcasted index comparison to constructthis mask, and you should take advantage of the fact that your scaled dot-product attention implementationfrom §3.5.4 already supports attention masking.

Applying RoPE. RoPE should be applied to the query and key vectors, but not the value vectors. Also,the head dimension should be handled as a batch dimension, because in multi-head attention, attention isbeing applied independently for each head. This means that precisely the same RoPE rotation should beapplied to the query and key vectors for each head.

Problem (multihead_self_attention): Implement causal multi-head self-attention (5points)

Deliverable: Implement causal multi-head self-attention as a torch.nn.Module. Your implemen-tation should accept (at least) the following parameters:

d_model: int Dimensionality of the Transformer block inputs.

num_heads: int Number of heads to use in multi-head self-attention.

Folllowing Vaswani et al. [2017], set $d _ { k } = d _ { v } = d _ { \mathrm { m o d e l } } / h$ . To test your implementation against ourprovided tests, implement the test adapter at [adapters.run_multihead_self_attention]. Then,run uv run pytest -k test_multihead_self_attention to test your implementation.

# 3.6 The Full Transformer LM

Let’s begin by assembling the Transformer block (it will be helpful to refer back to Figure 2). A Transformerblock contains two ‘sublayers’, one for the multihead self attention, and another for the feed-forward network.In each sublayer, we first perform RMSNorm, then the main operation (MHA/FF), finally adding in theresidual connection.

To be concrete, the first half (the first ‘sub-layer’) of the Transformer block should be implementing thefollowing set of updates to produce an output $y$ from an input $x$ ,

$$
y = x + \operatorname {M u l t i H e a d S e l f A t t e n t i o n} (\operatorname {R M S N o r m} (x)). \tag {15}
$$

Problem (transformer_block): Implement the Transformer block (3 points)

Implement the pre-norm Transformer block as described in §3.5 and illustrated in Figure 2. YourTransformer block should accept (at least) the following parameters.

d_model: int Dimensionality of the Transformer block inputs.

num_heads: int Number of heads to use in multi-head self-attention.

d_ff: int Dimensionality of the position-wise feed-forward inner layer.

To test your implementation, implement the adapter [adapters.run_transformer_block]. Thenrun uv run pytest -k test_transformer_block to test your implementation.

Deliverable: Transformer block code that passes the provided tests.

Now we put the blocks together, following the high level diagram in Figure 1. Follow our description ofthe embedding in Section 3.1.1, feed this into num_layers Transformer blocks, and then pass that into thethree output layers to obtain a distribution over the vocabulary.

# Problem (transformer_lm): Implementing the Transformer LM (3 points)

Time to put it all together! Implement the Transformer language model as described in §3.1and illustrated in Figure 1. At minimum, your implementation should accept all the aforementionedconstruction parameters for the Transformer block, as well as these additional parameters:

vocab_size: int The size of the vocabulary, necessary for determining the dimensionality of the tokenembedding matrix.

context_length: int The maximum context length, necessary for determining the dimensionality ofthe position embedding matrix.

num_layers: int The number of Transformer blocks to use.

To test your implementation against our provided tests, you will first need to implement the testadapter at [adapters.run_transformer_lm]. Then, run uv run pytest -k test_transformer_lmto test your implementation.

Deliverable: A Transformer LM module that passes the above tests.

Resource accounting. It is useful to be able to understand how the various parts of the Transformerconsume compute and memory. We will go through the steps to do some basic “FLOPs accounting.” Thevast majority of FLOPS in a Transformer are matrix multiplies, so our core approach is simple:

1. Write down all the matrix multiplies in a Transformer forward pass.

2. Convert each matrix multiply into FLOPs required.

For this second step, the following fact will be useful:

Rule: Given $A \in \mathbb { R } ^ { m \times n }$ and $B \in \mathbb { R } ^ { n \times p }$ , the matrix-matrix product $A B$ requires 2mnp FLOPs.

To see this, note that $( A B ) [ i , j ] = A [ i , : ] \cdot B [ : , j ]$ , and that this dot product requires $n$ additions and $n$multiplications ( $_ { z n }$ FLOPs). Then, since the matrix-matrix product $A B$ has $m \times p$ entries, the total numberof FLOPS is $( 2 n ) ( m p ) = 2 m n p$ .

Now, before you do the next problem, it can be helpful to go through each component of your Transformerblock and Transformer LM, and list out all the matrix multiplies and their associated FLOPs costs.

# Problem (transformer_accounting): Transformer LM resource accounting (5 points)

(a) Consider GPT-2 XL, which has the following configuration:

```txt
vocab_size : 50,257  
context_length : 1,024  
num_layers : 48  
d_model : 1,600
```

num_heads : 25

d_ff : 6,400

Suppose we constructed our model using this configuration. How many trainable parameterswould our model have? Assuming each parameter is represented using single-precision floatingpoint, how much memory is required to just load this model?

Deliverable: A one-to-two sentence response.

(b) Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shapedmodel. How many FLOPs do these matrix multiplies require in total? Assume that our inputsequence has context_length tokens.

Deliverable: A list of matrix multiplies (with descriptions), and the total number of FLOPsrequired.

(c) Based on your analysis above, which parts of the model require the most FLOPs?

Deliverable: A one-to-two sentence response.

(d) Repeat your analysis with GPT-2 small (12 layers, 768 d_model, 12 heads), GPT-2 medium (24layers, 1024 d_model, 16 heads), and GPT-2 large (36 layers, 1280 d_model, 20 heads). As themodel size increases, which parts of the Transformer LM take up proportionally more or less ofthe total FLOPs?

Deliverable: For each model, provide a breakdown of model components and its associatedFLOPs (as a proportion of the total FLOPs required for a forward pass). In addition, provide aone-to-two sentence description of how varying the model size changes the proportional FLOPsof each component.

(e) Take GPT-2 XL and increase the context length to 16,384. How does the total FLOPs for oneforward pass change? How do the relative contribution of FLOPs of the model componentschange?

Deliverable: A one-to-two sentence response.

# 4 Training a Transformer LM

We now have the steps to preprocess the data (via tokenizer) and the model (Transformer). What remainsis to build all of the code to support training. This consists of the following:

• Loss: we need to define the loss function (cross-entropy).

• Optimizer: we need to define the optimizer to minimize this loss (AdamW).

• Training loop: we need all the supporting infrastructure that loads data, saves checkpoints, andmanages training.

# 4.1 Cross-entropy loss

Recall that the Transformer language model defines a distribution $p _ { \theta } ( x _ { i + 1 } \ | \ x _ { 1 : i } )$ for each sequence $x$ oflength $m + 1$ and $i = 1 , \ldots , m$ . Given a training set $D$ consisting of sequences of length $m$ , we define thestandard cross-entropy (negative log-likelihood) loss function:

$$
\ell (\theta ; D) = \frac {1}{| D | m} \sum_ {x \in D} \sum_ {i = 1} ^ {m} - \log p _ {\theta} \left(x _ {i + 1} \mid x _ {1: i}\right). \tag {16}
$$

(Note that a single forward pass in the Transformer yields $p _ { \theta } ( x _ { i + 1 } \mid x _ { 1 : i } )$ for al l $i = 1 , \ldots , m$ .)

In particular, the Transformer computes logits $o _ { i } \in \mathbb { R } ^ { \mathsf { v o c a b \_ s 1 z e } }$ for each position $i$ , which results in:6

$$
p \left(x _ {i + 1} \mid x _ {1: i}\right) = \operatorname {s o f t m a x} \left(o _ {i}\right) \left[ x _ {i + 1} \right] = \frac {\exp \left(o _ {i} \left[ x _ {i + 1} \right]\right)}{\sum_ {a = 1} ^ {\text {v o c a b - s i z e}} \exp \left(o _ {i} [ a ]\right)}. \tag {17}
$$

The cross entropy loss is generally defined with respect to the vector of logits $o _ { i } \in \mathbb { R } ^ { \mathsf { v o c a b \_ s 1 z e } }$ and targetxi+1.7 $x _ { i + 1 }$

Implementing the cross entropy loss requires some care with numerical issues, just like in the case ofsoftmax.

# Problem (cross_entropy): Implement Cross entropy

Deliverable: Write a function to compute the cross entropy loss, which takes in predicted logits( $o _ { i }$ ) and targets $\left( x _ { i + 1 } \right)$ and computes the cross entropy $\ell _ { i } = - \log \operatorname { s o f t m a x } ( o _ { i } ) [ x _ { i + 1 } ]$ . Your functionshould handle the following:

• Subtract the largest element for numerical stability.

• Cancel out log and exp whenever possible.

• Handle any additional batch dimensions and return the average across the batch. As with sec-tion 3.3, we assume batch-like dimensions always come first, before the vocabulary size dimension.

Implement [adapters.run_cross_entropy], then run uv run pytest -k test_cross_entropyto test your implementation.

Perplexity Cross entropy suffices for training, but when we evaluate the model, we also want to reportperplexity. For a sequence of length $m$ where we suffer cross-entropy losses $\ell _ { 1 } , \ldots , \ell _ { m }$ :

$$
\text {p e r p l e x i t y} = \exp \left(\frac {1}{m} \sum_ {i = 1} ^ {m} \ell_ {i}\right). \tag {18}
$$

# 4.2 The SGD Optimizer

Now that we have a loss function, we will begin our exploration of optimizers. The simplest gradient-basedoptimizer is Stochastic Gradient Descent (SGD). We start with randomly initialized parameters $\theta _ { 0 }$ . Thenfor each step $t = 0 , \ldots , T - 1$ , we perform the following update:

$$
\theta_ {t + 1} \leftarrow \theta_ {t} - \alpha_ {t} \nabla L \left(\theta_ {t}; B _ {t}\right), \tag {19}
$$

where $B _ { t }$ is a random batch of data sampled from the dataset $D$ , and the learning rate $\alpha _ { t }$ and batch size$| B _ { t } |$ are hyperparameters.

# 4.2.1 Implementing SGD in PyTorch

To implement our optimizers, we will subclass the PyTorch torch.optim.Optimizer class. An Optimizersubclass must implement two methods:

def _init__(self, params, ...) should initialize your optimizer. Here, params will be a collection ofparameters to be optimized (or parameter groups, in case the user wants to use different hyperpa-rameters, such as learning rates, for different parts of the model). Make sure to pass params to the$\mathrm { ~ \tt ~ \_ { i n i t } ^ { \mathrm { ~ ~ } } ~ }$ method of the base class, which will store these parameters for use in step. You can takeadditional arguments depending on the optimizer (e.g., the learning rate is a common one), and passthem to the base class constructor as a dictionary, where keys are the names (strings) you choose forthese parameters.

def step(self) should make one update of the parameters. During the training loop, this will be calledafter the backward pass, so you have access to the gradients on the last batch. This method shoulditerate through each parameter tensor $\mathrm { \Delta p }$ and modify them in place, i.e. setting p.data, which holdsthe tensor associated with that parameter based on the gradient p.grad (if it exists), the tensorrepresenting the gradient of the loss with respect to that parameter.

The PyTorch optimizer API has a few subtleties, so it’s easier to explain it with an example. To makeour example richer, we’ll implement a slight variation of SGD where the learning rate decays over training,starting with an initial learning rate $\alpha$ and taking successively smaller steps over time:

$$
\theta_ {t + 1} = \theta_ {t} - \frac {\alpha}{\sqrt {t + 1}} \nabla L \left(\theta_ {t}; B _ {t}\right) \tag {20}
$$

Let’s see how this version of SGD would be implemented as a PyTorch Optimizer:

from collections.abc import Callable, iterable   
from typing import Optional   
import torch   
import math   
class SGD(torch optim.Optimizer): def__init__(self, params，lr=1e-3): if lr  $<  0$  ： raise ValueError(f"Invalid learning rate:{lr}") defaults  $=$  {"lr":lr} super(）.__init__(params，defaults) def step(self,closure:Optional[Callable]  $\equiv$  None): loss  $\equiv$  None if closure is None else closure() for group in self param_groups: lr  $=$  group["lr"] #Get the learning rate.

for p in group["params']: if p.grad is None: continue state  $=$  self.state[p] #Get state associated with  $p$  t  $=$  state.get("t",0) #Get iteration number from the state, or initial value. grad  $=$  p.grad.data #Get the gradient of loss with respect to  $p$  . p.data  $\equiv$  lr / math.sqrt(t + 1) \*grad #Update weight tensor in-place. state["t"]  $= t + 1$  #Increment iteration number.   
turn loss

In $\underline { { \mathbf { i n i t } } } _ { -- }$ , we pass the parameters to the optimizer, as well as default hyperparameters, to the baseclass constructor (the parameters might come in groups, each with different hyperparameters). In case theparameters are just a single collection of torch.nn.Parameter objects, the base constructor will create asingle group and assign it the default hyperparameters. Then, in step, we iterate over each parameter group,then over each parameter in that group, and apply Eq 20. Here, we keep the iteration number as a stateassociated with each parameter: we first read this value, use it in the gradient update, and then update it.The API specifies that the user might pass in a callable closure to re-compute the loss before the optimizerstep. We won’t need this for the optimizers we’ll use, but we add it to comply with the API.

To see this working, we can use the following minimal example of a training loop:

weights  $=$  torch.nn.Parameters(5 \* torch.random((10，10))) opt  $=$  SGD([weights],lr=1)

```python
for t in range(100):
    opt.zero_grad() # Reset the gradients for all learnable parameters.
    loss = (weights**2).mean() # Compute a scalar loss value.
    print(loss.cpu().item())
    loss.backup() # Run backward pass, which computes gradients.
    opt.step() # Run optimizer step.
```

This is the typical structure of a training loop: in each iteration, we will compute the loss and run astep of the optimizer. When training language models, our learnable parameters will come from the model(in PyTorch, m.parameters() gives us this collection). The loss will be computed over a sampled batch ofdata, but the basic structure of the training loop will be the same.

# Problem (learning_rate_tuning): Tuning the learning rate (1 point)

As we will see, one of the hyperparameters that affects training the most is the learning rate. Let’ssee that in practice in our toy example. Run the SGD example above with three other values for thelearning rate: 1e1, 1e2, and 1e3, for just 10 training iterations. What happens with the loss for eachof these learning rates? Does it decay faster, slower, or does it diverge (i.e., increase over the course oftraining)?

Deliverable: A one-two sentence response with the behaviors you observed.

# 4.3 AdamW

Modern language models are typically trained with more sophisticated optimizers, instead of SGD. Mostoptimizers used recently are derivatives of the Adam optimizer [Kingma and Ba, 2015]. We will use AdamW[Loshchilov and Hutter, 2019], which is in wide use in recent work. AdamW proposes a modification to Adamthat improves regularization by adding weight decay (at each iteration, we pull the parameters towards 0),

in a way that is decoupled from the gradient update. We will implement AdamW as described in algorithm2 of Loshchilov and Hutter [2019].

AdamW is stateful: for each parameter, it keeps track of a running estimate of its first and secondmoments. Thus, AdamW uses additional memory in exchange for improved stability and convergence.Besides the learning rate $\alpha$ , AdamW has a pair of hyperparameters $( \beta _ { 1 } , \beta _ { 2 } )$ that control the updates to themoment estimates, and a weight decay rate $\lambda$ . Typical applications set $( \beta _ { 1 } , \beta _ { 2 } )$ to (0.9, 0.999), but largelanguage models like LLaMA [Touvron et al., 2023] and GPT-3 [Brown et al., 2020] are often trained with(0.9, 0.95). The algorithm can be written as follows, where $\epsilon$ is a small value (e.g., $1 0 ^ { - 8 }$ ) used to improvenumerical stability in case we get extremely small values in $v$ :


Algorithm 1 AdamW Optimizer


init  $(\theta)$  (Initialize learnable parameters)   
 $m\gets 0$  (Initial value of the first moment vector; same shape as  $\theta$ $v\gets 0$  (Initial value of the second moment vector; same shape as  $\theta$    
for  $t = 1,\ldots ,T$  do Sample batch of data  $B_{t}$ $g\gets \nabla_{\theta}\ell (\theta ;B_t)$  (Compute the gradient of the loss at the current time step)  $m\gets \beta_1m + (1 - \beta_1)g$  (Update the first moment estimate)  $v\gets \beta_2v + (1 - \beta_2)g^2$  (Update the second moment estimate)  $\alpha_{t}\gets \alpha \frac{\sqrt{1 - (\beta_{2})^{t}}}{1 - (\beta_{1})^{t}}$  (Compute adjusted  $\alpha$  for iteration  $t$  1-  $(\beta_{1})^{t}$ $\theta \leftarrow \theta -\alpha_{t}\frac{m}{\sqrt{v} + \epsilon}$  (Update the parameters)  $\theta \gets \theta -\alpha \lambda \theta$  (Apply weight decay)   
end for

Note that $t$ starts at 1. You will now implement this optimizer.


Problem (adamw): Implement AdamW (2 points)


Deliverable: Implement the AdamW optimizer as a subclass of torch.optim.Optimizer. Yourclass should take the learning rate $\alpha$ in __init__, as well as the $\beta$ , $\epsilon$ and $\lambda$ hyperparameters. To helpyou keep state, the base Optimizer class gives you a dictionary self.state, which maps nn.Parameterobjects to a dictionary that stores any information you need for that parameter (for AdamW, this wouldbe the moment estimates). Implement [adapters.get_adamw_cls] and make sure it passes uv runpytest -k test_adamw.


Problem (adamwAccounting): Resource accounting for training with AdamW (2 points)


Let us compute how much memory and compute running AdamW requires. Assume we are usingfloat32 for every tensor.

(a) How much peak memory does running AdamW require? Decompose your answer based on thememory usage of the parameters, activations, gradients, and optimizer state. Express your answerin terms of the batch_size and the model hyperparameters (vocab_size, context_length,num_layers, d_model, num_heads). Assume d_ff = 4 × d_model.

For simplicity, when calculating memory usage of activations, consider only the following compo-nents:

• Transformer block

– RMSNorm(s)

– Multi-head self-attention sublayer: $Q K V$ projections, $Q ^ { \mid } K$ matrix multiply, softmax,weighted sum of values, output projection.

– Position-wise feed-forward: $W _ { 1 }$ matrix multiply, SiLU, $W _ { 2 }$ matrix multiply

• final RMSNorm

• output embedding

• cross-entropy on logits

Deliverable: An algebraic expression for each of parameters, activations, gradients, and opti-mizer state, as well as the total.

(b) Instantiate your answer for a GPT-2 XL-shaped model to get an expression that only depends onthe batch_size. What is the maximum batch size you can use and still fit within 80GB memory?

Deliverable: An expression that looks like a · batch_size + b for numerical values $a , b$ , and anumber representing the maximum batch size.

(c) How many FLOPs does running one step of AdamW take?

Deliverable: An algebraic expression, with a brief justification.

(d) Model FLOPs utilization (MFU) is defined as the ratio of observed throughput (tokens per second)relative to the hardware’s theoretical peak FLOP throughput [Chowdhery et al., 2022]. AnNVIDIA A100 GPU has a theoretical peak of 19.5 teraFLOP/s for float32 operations. Assumingyou are able to get 50% MFU, how long would it take to train a GPT-2 XL for 400K steps and abatch size of 1024 on a single A100? Following Kaplan et al. [2020] and Hoffmann et al. [2022],assume that the backward pass has twice the FLOPs of the forward pass.

Deliverable: The number of days training would take, with a brief justification.

# 4.4 Learning rate scheduling

The value for the learning rate that leads to the quickest decrease in loss often varies during training. Intraining Transformers, it is typical to use a learning rate schedule, where we start with a bigger learningrate, making quicker updates in the beginning, and slowly decay it to a smaller value as the model trains8In this assignment, we will implement the cosine annealing schedule used to train LLaMA [Touvron et al.,2023].

A scheduler is simply a function that takes the current step $t$ and other relevant parameters (such as theinitial and final learning rates), and returns the learning rate to use for the gradient update at step $t$ . Thesimplest schedule is the constant function, which will return the same learning rate given any $t$ .

The cosine annealing learning rate schedule takes (i) the current iteration $t$ , (ii) the maximum learningrate $\alpha _ { \mathrm { m a x } }$ , (iii) the minimum (final) learning rate $\alpha _ { \mathrm { { m i n } } }$ , (iv) the number of warm-up iterations $T _ { w }$ , and (v)the number of cosine annealing iterations $T _ { c }$ . The learning rate at iteration $t$ is defined as:

(Warm-up) If $t < T _ { w }$ , then $\begin{array} { r } { \alpha _ { t } = \frac { t } { T _ { w } } \alpha _ { \mathrm { m a x } } } \end{array}$ Tw t αmax.

(Cosine annealing) If $T _ { w } \le t \le T _ { c }$ , then $\begin{array} { r } { \alpha _ { t } = \alpha _ { \operatorname* { m i n } } + \frac { 1 } { 2 } \left( 1 + \cos \left( \frac { t - T _ { w } } { T _ { c } - T _ { w } } \pi \right) \right) ( \alpha _ { \operatorname* { m a x } } - \alpha _ { \operatorname* { m i n } } ) . } \end{array}$

(Post-annealing) If $t > T _ { c }$ , then $\alpha _ { t } = \alpha _ { \operatorname* { m i n } }$

# Problem (learning_rate_schedule): Implement cosine learning rate schedule withwarmup

Write a function that takes $t$ , $\alpha _ { \mathrm { m a x } }$ , $\alpha _ { \mathrm { { m i n } } }$ , $T _ { w }$ and $T _ { c }$ , and returns the learning rate $\alpha _ { t }$ according tothe scheduler defined above. Then implement [adapters.get_lr_cosine_schedule] and make sureit passes uv run pytest -k test_get_lr_cosine_schedule.

# 4.5 Gradient clipping

During training, we can sometimes hit training examples that yield large gradients, which can destabilizetraining. To mitigate this, one technique often employed in practice is gradient clipping. The idea is toenforce a limit on the norm of the gradient after each backward pass before taking an optimizer step.

Given the gradient (for all parameters) $g$ , we compute its $\ell _ { 2 }$ -norm $\| g \| _ { 2 }$ . If this norm is less than amaximum value $M$ , then we leave $g$ as is; otherwise, we scale $g$ down by a factor of $\frac { M } { \lVert g \rVert _ { 2 } + \epsilon }$ (where a small $\epsilon$ ,like $1 0 ^ { - 6 }$ , is added for numeric stability). Note that the resulting norm will be just under $M$ .

# Problem (gradient_clipping): Implement gradient clipping (1 point)

Write a function that implements gradient clipping. Your function should take a list of parametersand a maximum $\ell _ { 2 }$ -norm. It should modify each parameter gradient in place. Use $\epsilon = 1 0 ^ { - 6 }$ (thePyTorch default). Then, implement the adapter [adapters.run_gradient_clipping] and make sureit passes uv run pytest -k test_gradient_clipping.

# 5 Training loop

We will now finally put together the major components we’ve built so far: the tokenized data, the model,and the optimizer.

# 5.1 Data Loader

The tokenized data (e.g., that you prepared in tokenizer_experiments) is a single sequence of tokens$x = ( x _ { 1 } , \ldots , x _ { n } ) $ . Even though the source data might consist of separate documents (e.g., different webpages, or source code files), a common practice is to concatenate all of those into a single sequence of tokens,adding a delimiter between them (such as the <|endoftext|> token).

A data loader turns this into a stream of batches, where each batch consists of $B$ sequences of length$m$ , paired with the corresponding next tokens, also with length $m$ . For example, for $B = 1 , m = 3$ ,$( [ x _ { 2 } , x _ { 3 } , x _ { 4 } ] , [ x _ { 3 } , x _ { 4 } , x _ { 5 } ] )$ would be one potential batch.

Loading data in this way simplifies training for a number of reasons. First, any $1 \leq i < n - m$ gives avalid training sequence, so sampling sequences are trivial. Since all training sequences have the same length,there’s no need to pad input sequences, which improves hardware utilization (also by increasing batch size$B$ ). Finally, we also don’t need to fully load the full dataset to sample training data, making it easy tohandle large datasets that might not otherwise fit in memory.

# Problem (data_loading): Implement data loading (2 points)

Deliverable: Write a function that takes a numpy array $x$ (integer array with token IDs), abatch_size, a context_length and a PyTorch device string (e.g., 'cpu' or 'cuda:0'), and returnsa pair of tensors: the sampled input sequences and the corresponding next-token targets. Both ten-sors should have shape (batch_size, context_length) containing token IDs, and both should beplaced on the requested device. To test your implementation against our provided tests, you will firstneed to implement the test adapter at [adapters.run_get_batch]. Then, run uv run pytest -ktest_get_batch to test your implementation.

# Low-Resource/Downscaling Tip: Data loading on CPU or Apple Silicon

If you are planning to train your LM on CPU or Apple Silicon, you need to move your datato the correct device (and similarly, you should use the same device for your model later on).

If you are on CPU, you can use the 'cpu' device string, and on Apple Silicon $\mathrm { ~ M ^ { * } ~ }$ chips), youcan use the 'mps' device string.

For more on MPS, checkout these resources:

• https://developer.apple.com/metal/pytorch/

• https://pytorch.org/docs/main/notes/mps.html

What if the dataset is too big to load into memory? We can use a Unix systemcall named mmap whichmaps a file on disk to virtual memory, and lazily loads the file contents when that memory location isaccessed. Thus, you can “pretend” you have the entire dataset in memory. Numpy implements this throughnp.memmap (or the flag mmap_mode='r' to np.load, if you originally saved the array with np.save), whichwill return a numpy array-like object that loads the entries on-demand as you access them. When samplingfrom your dataset (i.e., a numpy array) during training, be sure load the dataset in memory-mapped mode (via np.memmap or the flag mmap_mode='r' to np.load, depending on how you saved thearray). Make sure you also specify a dtype that matches the array that you’re loading. It may be helpfulto explicitly verify that the memory-mapped data looks correct (e.g., doesn’t contain values beyond theexpected vocabulary size).

# 5.2 Checkpointing

In addition to loading data, we will also need to save models as we train. When running jobs, we oftenwant to be able to resume a training run that for some reason stopped midway (e.g., due to your job timingout, machine failure, etc). Even when all goes well, we might also want to later have access to intermediatemodels (e.g., to study training dynamics post-hoc, take samples from models at different stages of training,etc).

A checkpoint should have all the states that we need to resume training. We of course want to be ableto restore model weights at a minimum. If using a stateful optimizer (such as AdamW), we will also needto save the optimizer’s state (e.g., in the case of AdamW, the moment estimates). Finally, to resume thelearning rate schedule, we will need to know the iteration number we stopped at. PyTorch makes it easy tosave all of these: every nn.Module has a state_dict() method that returns a dictionary with all learnableweights; we can restore these weights later with the sister method load_state_dict(). The same goesfor any nn.optim.Optimizer. Finally, torch.save(obj, dest) can dump an object (e.g., a dictionarycontaining tensors in some values, but also regular Python objects like integers) to a file (path) or file-likeobject, which can then be loaded back into memory with torch.load(src).

# Problem (checkpointing): Implement model checkpointing (1 point)

Implement the following two functions to load and save checkpoints:

def save_checkpoint(model, optimizer, iteration, out) should dump all the state from thefirst three parameters into the file-like object out. You can use the state_dict method of boththe model and the optimizer to get their relevant states and use torch.save(obj, out) to dumpobj into out (PyTorch supports either a path or a file-like object here). A typical choice is tohave obj be a dictionary, but you can use whatever format you want as long as you can load yourcheckpoint later.

This function expects the following parameters:

model: torch.nn.Module

optimizer: torch.optim.Optimizer

iteration: int

out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]

def load_checkpoint(src, model, optimizer) should load a checkpoint from src (path or file-like object), and then recover the model and optimizer states from that checkpoint. Yourfunction should return the iteration number that was saved to the checkpoint. You can usetorch.load(src) to recover what you saved in your save_checkpoint implementation, and theload_state_dict method in both the model and optimizers to return them to their previousstates.

This function expects the following parameters:

src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]

model: torch.nn.Module

optimizer: torch.optim.Optimizer

Implement the [adapters.run_save_checkpoint] and [adapters.run_load_checkpoint]adapters, and make sure they pass uv run pytest -k test_checkpointing.

# 5.3 Training loop

Now, it’s finally time to put all of the components you implemented together into your main training script.It will pay off to make it easy to start training runs with different hyperparameters (e.g., by taking themas command-line arguments), since you will be doing these many times later to study how different choicesimpact training.

# Problem (training_together): Put it together (4 points)

Deliverable: Write a script that runs a training loop to train your model on user-provided input.In particular, we recommend that your training script allow for (at least) the following:

• Ability to configure and control the various model and optimizer hyperparameters.

• Memory-efficient loading of training and validation large datasets with np.memmap.

• Serializing checkpoints to a user-provided path.

• Periodically logging training and validation performance (e.g., to console and/or an externalservice like Weights and Biases).a

# 6 Generating text

Now that we can train models, the last piece we need is the ability to generate text from our model.Recall that a language model takes in a (possibly batched) integer sequence of length (sequence_length)and produces a matrix of size (sequence_length $\times$ vocab size), where each element of the sequence is aprobability distribution predicting the next word after that position. We will now write a few functions toturn this into a sampling scheme for new sequences.

Softmax By standard convention, the language model output is the output of the final linear layer (the“logits”) and so we have to turn this into a normalized probability via the softmax operation, which we sawearlier in Eq 10.

Decoding To generate text (decode) from our model, we will provide the model with a sequence of prefixtokens (the “prompt”), and ask it to produce a probability distribution over the vocabulary that predictsthe next word in the sequence. Then, we will sample from this distribution over the vocabulary items todetermine the next output token.

Concretely, one step of the decoding process should take in a sequence $x _ { 1 \ldots t }$ and return a token $x _ { t + 1 }$ viathe following equation,

$$
\begin{array}{l} P (x _ {t + 1} = i \mid x _ {1 \dots t}) = \frac {\exp (v _ {i})}{\sum_ {j} \exp (v _ {j})} \\ v = \operatorname {T r a n s f o r m e r L M} (x _ {1 \dots t}) _ {t} \in \mathbb {R} ^ {\text {v o c a b - s i z e}} \\ \end{array}
$$

where TransformerLM is our model which takes as input a sequence of sequence_length and produces amatrix of size (sequence_length $\times$ vocab_size), and we take the last element of this matrix, as we arelooking for the next word prediction at the $t$ -th position.

This gives us a basic decoder by repeatedly sampling from these one-step conditionals (appending ourpreviously-generated output token to the input of the next decoding timestep) until we generate the end-of-sequence token <|endoftext|> (or a user-specified maximum number of tokens to generate).

Decoder tricks We will be experimenting with small models, and small models can sometimes generatevery low quality texts. Two simple decoder tricks can help fix these issues. First, in temperature scaling wemodify our softmax with a temperature parameter $\tau$ , where the new softmax is

$$
\operatorname {s o f t m a x} (v, \tau) _ {i} = \frac {\exp \left(v _ {i} / \tau\right)}{\sum_ {j = 1} ^ {\left| \text {v o c a b ＿ s i z e} \right|} \exp \left(v _ {j} / \tau\right)}. \tag {24}
$$

Note how setting $\tau  0$ makes it so that the largest element of $\boldsymbol { v }$ dominates, and the output of the softmaxbecomes a one-hot vector concentrated at this maximal element.

Second, another trick is nucleus or top- $p$ sampling, where we modify the sampling distribution by trun-cating low-probability words. Let $q$ be a probability distribution that we get from a (temperature-scaled)softmax of size (vocab_size). Nucleus sampling with hyperparameter $p$ produces the next token accordingto the equation

$$
P (x _ {t + 1} = i | q) = \left\{ \begin{array}{l l} \frac {q _ {i}}{\sum_ {j \in V (p)} q _ {j}} & \text {i f} i \in V (p) \\ 0 & \text {o t h e r w i s e} \end{array} \right.
$$

where $V ( p )$ is the smallest set of indices such that $\Sigma _ { j \in V ( p ) } q _ { j } \geq p$ . You can compute this quantity easily byfirst sorting the probability distribution $q$ by magnitude, and selecting the largest vocabulary elements untilyou reach the target level of $\alpha$ .

# Problem (decoding): Decoding (3 points)

Deliverable: Implement a function to decode from your language model. We recommend that yousupport the following features:

• Generate completions for a user-provided prompt (i.e., take in some $x _ { 1 \dots t }$ and sample a completionuntil you hit an <|endoftext|> token).

• Allow the user to control the maximum number of generated tokens.

• Given a desired temperature value, apply softmax temperature scaling to the predicted next-worddistributions before sampling.

• Top-p sampling (Holtzman et al., 2020; also referred to as nucleus sampling), given a user-specifiedthreshold value.

# 7 Experiments

Now it is time to put everything together and train (small) language models on a pretaining dataset.

# 7.1 How to Run Experiments and Deliverables

The best way to understand the rationale behind the architectural components of a Transformer is to actuallymodify it and run it yourself. There is no substitute for hands-on experience.

To this end, it’s important to be able to experiment quickly, consistently, and keep records ofwhat you did. To experiment quickly, we will be running many experiments on a small scale model (17Mparameters) and simple dataset (TinyStories). To do things consistently, you will ablate components andvary hyperparameters in a systematic way, and to keep records we will ask you to submit a log of yourexperiments and learning curves associated with each experiment.

To make it possible to submit loss curves, make sure to periodically evaluate validation lossesand record both the number of steps and wallclock times. You might find logging infrastructuresuch as Weights and Biases helpful.

# Problem (experiment_log): Experiment logging (3 points)

For your training and evaluation code, create experiment tracking infrastructure that allows you totrack your experiments and loss curves with respect to gradient steps and wallclock time.

Deliverable: Logging infrastructure code for your experiments and an experiment log (a documentof all the things you tried) for the assignment problems below in this section.

# 7.2 TinyStories

We are going to start with a very simple dataset (TinyStories; Eldan and Li, 2023) where models will trainquickly, and we can see some interesting behaviors. The instructions for getting this dataset is at section 1.An example of what this dataset looks like is below.

# Example (tinystories_example): One example from TinyStories

Once upon a time there was a little boy named Ben. Ben loved to explore the world around him.He saw many amazing things, like beautiful vases that were on display in a store. One day, Ben waswalking through the store when he came across a very special vase. When Ben saw it he was amazed!He said, “Wow, that is a really amazing vase! Can I buy it?” The shopkeeper smiled and said, “Ofcourse you can. You can take it home and show all your friends how amazing it is!” So Ben took thevase home and he was so proud of it! He called his friends over and showed them the amazing vase.All his friends thought the vase was beautiful and couldn’t believe how lucky Ben was. And that’s howBen found an amazing vase in the store!

Hyperparameter tuning We will tell you some very basic hyperparameters to start with and ask you tofind some settings for others that work well.

vocab_size 10000. Typical vocabulary sizes are in the tens to hundreds of thousands. You should vary thisand see how the vocabulary and model behavior changes.

context_length 256. Simple datasets such as TinyStories might not need long sequence lengths, but forthe later OpenWebText data, you may want to vary this. Try varying this and seeing the impact onboth the per-iteration runtime and the final perplexity.

d_model 512. This is slightly smaller than the 768 dimensions used in many small Transformer papers, butthis will make things faster.

d_ff 1344. This is roughly ${ \frac { 8 } { 3 } } \mathsf { d } _ { \cdot }$ _model while being a multiple of 64, which is good for GPU performance.

RoPE theta parameter Θ 10000.

number of layers and heads 4 layers, 16 heads. Together, this will give about 17M non-embedding pa-rameters which is a fairly small Transformer.

total tokens processed 327,680,000 (your batch size $\times$ total step count $\times$ context length should equalroughly this value).

You should do some trial and error to find good defaults for the following other hyperparameters:learning rate, learning rate warmup, other AdamW hyperparameters $( \beta _ { 1 } , \beta _ { 2 } , \epsilon )$ , and weight decay.You can find some typical choices of such hyperparameters in Kingma and Ba [2015].

Putting it together Now you can put everything together by getting a trained BPE tokenizer, tok-enizing the training dataset, and running this in the training loop that you wrote. Important note: Ifyour implementation is correct and efficient, the above hyperparameters should result in a roughly 30-40minute runtime on 1 H100 GPU. If you have runtimes that are much longer, please check and make sureyour dataloading, checkpointing, or validation loss code is not bottlenecking your runtimes and that yourimplementation is properly batched.

Tips and tricks for debugging model architectures We highly recommend getting comfortable withyour IDE’s built-in debugger (e.g., VSCode/PyCharm), which will save you time compared to debuggingwith print statements. If you use a text editor, you can use something more like pdb. A few other goodpractices when debugging model architectures are:

• A common first step when developing any neural net architecture is to overfit to a single minibatch. Ifyour implementation is correct, you should be able to quickly drive the training loss to near-zero.

• Set debug breakpoints in various model components, and inspect the shapes of intermediate tensors tomake sure they match your expectations.

• Monitor the norms of activations, model weights, and gradients to make sure they are not explodingor vanishing.

# Problem (learning_rate): Tune the learning rate (3 points) (4 H100 hrs)

The learning rate is one of the most important hyperparameters to tune. Taking the base modelyou’ve trained, answer the following questions:

(a) Perform a hyperparameter sweep over the learning rates and report the final losses (or notedivergence if the optimizer diverges).

Deliverable: Learning curves associated with multiple learning rates. Explain your hyperpa-rameter search strategy.

Deliverable: A model with validation loss (per-token) on TinyStories of at most 1.45

# Low-Resource/Downscaling Tip: Train for few steps on CPU or Apple Silicon

If you are running on cpu or mps, you should instead reduce the total tokens processedcount to 40, 000, 000, which will be sufficient to produce reasonably fluent text. You mayalso increase the target validation loss from 1.45 to 2.00.

Running our solution code with a tuned learning rate on an M3 Max chip and 36 GB ofRAM, we use batch size $\times$ total step count $\times$ context length = 32×5000×256 = 40, 960, 000tokens, which takes 1 hour and 22 minutes on cpu and 36 minutes on mps. At step 5000,we achieve a validation loss of 1.80.

Some additional tips:

• When using $X$ training steps, we suggest adjusting the cosine learning rate decayschedule to terminate its decay (i.e., reach the minimum learning rate) at preciselystep $X$ .

• When using mps, do not use TF32 kernels, i.e., do not settorch.set_float32_matmul_precision('high')

as you might with cuda devices. We tried enabling TF32 kernels with mps (torchversion 2.6.0) and found the backend will use silently broken kernels that cause unstabletraining.

• You can speed up training by JIT-compiling your model with torch.compile. Specif-ically:

– On cpu, compile your model with

$$
\text {m o d e l} = \text {t o r c h . c o m p i l e (m o d e l)}
$$

– On mps, you can somewhat optimize the backward pass using

$$
\text {m o d e l} = \text {t o r c h . c o m p i l e (m o d e l , b a c k e n d} = ^ {\prime \prime} a o t _ {e a g e r} ^ {\prime \prime})
$$

Compilation with Inductor is not supported on mps as of torch version 2.6.0.

(b) Folk wisdom is that the best learning rate is “at the edge of stability.” Investigate how the pointat which learning rates diverge is related to your best learning rate.

Deliverable: Learning curves of increasing learning rate which include at least one divergentrun and an analysis of how this relates to convergence rates.

Now let’s vary the batch size and see what happens to training. Batch sizes are important – they let us gethigher efficiency from our GPUs by doing larger matrix multiplies, but is it true that we always want batchsizes to be large? Let’s run some experiments to find out.

# Problem (batch_size_experiment): Batch size variations (1 point) (2 H100 hrs)

Vary your batch size all the way from 1 to the GPU memory limit. Try at least a few batch sizesin between, including typical sizes like 64 and 128.

Deliverable: Learning curves for runs with different batch sizes. The learning rates should beoptimized again if necessary.

Deliverable: A few sentences discussing of your findings on batch sizes and their impacts ontraining.

With your decoder in hand, we can now generate text! We will generate from the model and see howgood it is. As a reference, you should get outputs that look at least as good as the example below.

# Example (ts_generate_example): Sample output from a TinyStories language model

Once upon a time, there was a pretty girl named Lily. She loved to eat gum, especially the big blackone. One day, Lily’s mom asked her to help cook dinner. Lily was so excited! She loved to help hermom. Lily’s mom made a big pot of soup for dinner. Lily was so happy and said, “Thank you, Mommy!I love you.” She helped her mom pour the soup into a big bowl. After dinner, Lily’s mom made someyummy soup. Lily loved it! She said, “Thank you, Mommy! This soup is so yummy!” Her mom smiledand said, “I’m glad you like it, Lily.” They finished cooking and continued to cook together. The end.

# Low-Resource/Downscaling Tip: Generate text on CPU or Apple Silicon

If instead you used the low-resource configuration with 40M tokens processed, you should see gen-erations that still resemble English but are not as fluent as above. For example, our sample outputfrom a TinyStories language model trained on 40M tokens is below:

Once upon a time, there was a little girl named Sue. Sue had a tooth that she loved very much. Itwas his best head. One day, Sue went for a walk and met a ladybug! They became good friends andplayed on the path together.

“Hey, Polly! Let’s go out!” said Tim. Sue looked at the sky and saw that it was difficult to find away to dance shining. She smiled and agreed to help the talking!”

As Sue watched the sky moved, what it was. She

Here is the precise problem statement and what we ask for:

# Problem (generate): Generate text (1 point)

Using your decoder and your trained checkpoint, report the text generated by your model. Youmay need to manipulate decoder parameters (temperature, top-p, etc.) to get fluent outputs.

Deliverable: Text dump of at least 256 tokens of text (or until the first <|endoftext|> token),and a brief comment on the fluency of this output and at least two factors which affect how good orbad this output is.

# 7.3 Ablations and architecture modification

The best way to understand the Transformer is to actually modify it and see how it behaves. We will nowdo a few simple ablations and modifications.

Ablation 1: layer normalization It is often said that layer normalization is important for the stabilityof Transformer training. But perhaps we want to live dangerously. Let’s remove RMSNorm from each ofour Transformer blocks and see what happens.

# Problem (layer_norm_ablation): Remove RMSNorm and train (1 point) (1 H100 hr)

Remove all of the RMSNorms from your Transformer and train. What happens at the previousoptimal learning rate? Can you get stability by using a lower learning rate?

Deliverable: A learning curve for when you remove RMSNorms and train, as well as a learningcurve for the best learning rate.

Deliverable: A few sentence commentary on the impact of RMSNorm.

Let’s now investigate another layer normalization choice that seems arbitrary at first glance. Pre-normTransformer blocks are defined as

$$
z = x + \text {M u l t i H e a d e d S e l f A t t e n t i o n} (\operatorname {R M S N o r m} (x))
$$

$$
y = z + \operatorname {F F N} (\operatorname {R M S N o r m} (z)).
$$

This is one of the few ‘consensus’ modifications to the original Transformer architecture, which used apost-norm approach as

$$
z = \operatorname {R M S N o r m} (x + \text {M u l t i H e a d e d S e l f A t t e n t i o n} (x))
$$

$$
y = \operatorname {R M S N o r m} (z + \operatorname {F F N} (z)).
$$

Let’s revert back to the post-norm approach and see what happens.

# Problem (pre_norm_ablation): Implement post-norm and train (1 point) (1 H100 hr)

Modify your pre-norm Transformer implementation into a post-norm one. Train with the post-normmodel and see what happens.

Deliverable: A learning curve for a post-norm transformer, compared to the pre-norm one.

We see that layer normalization has a major impact on the behavior of the transformer, and that eventhe position of the layer normalization is important.

Ablation 2: position embeddings We will next investigate the impact of the position embeddings onthe performance of the model. Specifically, we will compare our base model (with RoPE) with not includingposition embeddings at all (NoPE). It turns out that decoder-only transformers, i.e., those with a causalmask as we have implemented, can in theory infer relative or absolute position information without beingprovided with position embeddings explicitly [Tsai et al., 2019, Kazemnejad et al., 2023]. We will now testempirically how NoPE performs compare to RoPE.

# Problem (no_pos_emb): Implement NoPE (1 point) (1 H100 hr)

Modify your Transformer implementation with RoPE to remove the position embedding informationentirely, and see what happens.

Deliverable: A learning curve comparing the performance of RoPE and NoPE.

Ablation 3: SwiGLU vs. SiLU Next, we will follow Shazeer [2020] and test the importance of gatingin the feed-forward network, by comparing the performance of SwiGLU feed-forward networks versus feed-forward networks using SiLU activations but no gated linear unit (GLU):

$$
\mathrm {F F N} _ {\mathrm {S i L U}} (x) = W _ {2} \mathrm {S i L U} \left(W _ {1} x\right). \tag {25}
$$

Recall that in our SwiGLU implementation, we set the dimensionality of the inner feed-forward layer tobe roughly $d _ { \mathrm { f f } } = \frac { 8 } { 3 } d _ { \mathrm { m o d e l } }$ (while ensuring that $d _ { \mathrm { f f } }$ mod $6 4 = 0$ , to make use of GPU tensor cores). In yourFFNSiLU implementation you should set $d _ { \mathrm { f f } } = 4 \times d _ { \mathrm { m o d e l } }$ , to approximately match the parameter count ofthe SwiGLU feed-forward network (which has three instead of two weight matrices).

# Problem (swiglu_ablation): SwiGLU vs. SiLU (1 point) (1 H100 hr)

Deliverable: A learning curve comparing the performance of SwiGLU and SiLU feed-forwardnetworks, with approximately matched parameter counts.

# Low-Resource/Downscaling Tip: Online students with limited GPU resources shouldtest modifications on TinyStories

In the remainder of the assignment, we will move to a larger-scale, noisier web dataset (Open-WebText), experimenting with architecture modifications and (optionally) making a submission to thecourse leaderboard.

It takes a long time to train an LM to fluency on OpenWebText, so we suggest that online studentswith limited GPU access continue testing modifications on TinyStories (using validation loss as a metricto evaluate performance).

# 7.4 Running on OpenWebText

We will now move to a more standard pretraining dataset created from a webcrawl. A small sample ofOpenWebText [Gokaslan et al., 2019] is also provided as a single text file: see section 1 for how to accessthis file.

Here is an example from OpenWebText. Note how the text is much more realistic, complex, and varied.You may want to look through the training dataset to get a sense of what training data looks like for awebscraped corpus.

# Example (owt_example): One example from OWT

Baseball Prospectus director of technology Harry Pavlidis took a risk when he hired Jonathan Judge.

Pavlidis knew that, as Alan Schwarz wrote in The Numbers Game, “no corner of American cultureis more precisely counted, more passionately quantified, than performances of baseball players.” Witha few clicks here and there, you can findout that Noah Syndergaard’s fastball revolves more than 2,100times per minute on its way to the plate, that Nelson Cruz had the game’s highest average exit velocityamong qualified hitters in 2016 and myriad other tidbits that seem ripped from a video game or sciencefiction novel. The rising ocean of data has empowered an increasingly important actor in baseball’sculture: the analytical hobbyist.

That empowerment comes with added scrutiny – on the measurements, but also on the peopleand publications behind them. With Baseball Prospectus, Pavlidis knew all about the backlash thataccompanies quantitative imperfection. He also knew the site’s catching metrics needed to be reworked,and that it would take a learned mind – someone who could tackle complex statistical modeling problems– to complete the job.

“He freaks us out.” Harry Pavlidis

Pavlidis had a hunch that Judge “got it” based on the latter’s writing and their interaction at a site-sponsored ballpark event. Soon thereafter, the two talked over drinks. Pavlidis’ intuition was validated.Judge was a fit for the position – better yet, he was a willing fit. “I spoke to a lot of people,” Pavlidissaid, “he was the only one brave enough to take it on.” [...]

Note: You may have to re-tune your hyperparameters such as learning rate or batch size for this experiment.

# Problem (main_experiment): Experiment on OWT (2 points) (3 H100 hrs)

Train your language model on OpenWebText with the same model architecture and total trainingiterations as TinyStories. How well does this model do?

Deliverable: A learning curve of your language model on OpenWebText. Describe the differencein losses from TinyStories – how should we interpret these losses?

Deliverable: Generated text from OpenWebText LM, in the same format as the TinyStoriesoutputs. How is the fluency of this text? Why is the output quality worse even though we have thesame model and compute budget as TinyStories?

# 7.5 Your own modification $+$ leaderboard

Congratulations on getting to this point. You’re almost done! You will now try to improve upon theTransformer architecture, and see how your hyperparameters and architecture stack up against other studentsin the class.

Rules for the leaderboard There are no restrictions other than the following:

Runtime Your submission can run for at most 1.5 hours on an H100. You can enforce this by setting--time=01:30:00 in your slurm submission script.

Data You may only use the OpenWebText training dataset that we provide.

Otherwise, you are free to do whatever your heart desires.

If you are looking for some ideas on what to implement, you can checkout some of these resources:

• State-of-the-art open-source LLM families, such as Llama 3 [Grattafiori et al., 2024] or Qwen 2.5 [Yanget al., 2024].

• The NanoGPT speedrun repository (https://github.com/KellerJordan/modded-nanogpt), wherecommunity members post many interesting modifications for “speedrunning” small-scale languagemodel pretraining. For example, a common modification that dates back to the original Transformerpaper is to tie the weights of the input and output embeddings together (see Vaswani et al. [2017](Section 3.4) and Chowdhery et al. [2022] (Section 2)). If you do try weight tying, you may have todecrease the standard deviation of the embedding/LM head init.

You will want to test these on either a small subset of OpenWebText or on TinyStories before trying thefull 1.5-hour run.

As a caveat, we do note that some of the modifications you may find working well in this leaderboardmay not generalize to larger-scale pretraining. We will explore this idea further in the scaling laws unit ofthe course.

# Problem (leaderboard): Leaderboard (6 points) (10 H100 hrs)

You will train a model under the leaderboard rules above with the goal of minimizing the validationloss of your language model within 1.5 H100-hour.

Deliverable: The final validation loss that was recorded, an associated learning curve that clearlyshows a wallclock-time x-axis that is less than 1.5 hours and a description of what you did. We expecta leaderboard submission to beat at least the naive baseline of a 5.0 loss. Submit to the leaderboardhere: https://github.com/stanford-cs336/assignment1-basics-leaderboard.

# References



Ronen Eldan and Yuanzhi Li. TinyStories: How small can language models be and still speak coherentEnglish?, 2023. arXiv:2305.07759.





Aaron Gokaslan, Vanya Cohen, Ellie Pavlick, and Stefanie Tellex. OpenWebText corpus. http://Skylion007.github.io/OpenWebTextCorpus, 2019.





Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words with subwordunits. In Proc. of ACL, 2016.





Changhan Wang, Kyunghyun Cho, and Jiatao Gu. Neural machine translation with byte-level subwords,2019. arXiv:1909.03341.





Philip Gage. A new algorithm for data compression. C Users Journal, 12(2):23–38, February 1994. ISSN0898-9788.





Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models areunsupervised multitask learners, 2019.





Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understandingby generative pre-training, 2018.





Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser,and Illia Polosukhin. Attention is all you need. In Proc. of NeurIPS, 2017.





Toan Q. Nguyen and Julian Salazar. Transformers without tears: Improving the normalization of self-attention. In Proc. of IWSWLT, 2019.





Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan,Liwei Wang, and Tie-Yan Liu. On layer normalization in the Transformer architecture. In Proc. of ICML,2020.





Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer normalization, 2016. arXiv:1607.06450.





Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix,Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin,Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models, 2023.arXiv:2302.13971.





Biao Zhang and Rico Sennrich. Root mean square layer normalization. In Proc. of NeurIPS, 2019.





Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, AnirudhGoyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, ArthurHinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere,Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra,Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian CantonFerrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, Danny Wyatt,David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes,Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic,Francisco Guzmán, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, GovindThattai, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar,Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan Misra, Ivan Evti-mov, Jack Zhang, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, JeetShah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu



Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, JoshuaJohnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Karthik Prasad, Kartikeya Upasani, KatePlawiak, Ke Li, Kenneth Heafield, Kevin Stone, Khalid El-Arini, Krithika Iyer, Kshitiz Malik, KuenleyChiu, Kunal Bhalla, Kushal Lakhotia, Lauren Rantala-Yeary, Laurens van der Maaten, Lawrence Chen,Liang Tan, Liz Jenkins, Louis Martin, Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Lukede Oliveira, Madeline Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin Kardas, MariaTsimpoukelli, Mathew Oldham, Mathieu Rita, Maya Pavlova, Melanie Kambadur, Mike Lewis, Min Si,Mitesh Kumar Singh, Mona Hassan, Naman Goyal, Narjes Torabi, Nikolay Bashlykov, Nikolay Bogoychev,Niladri Chatterji, Ning Zhang, Olivier Duchenne, Onur Çelebi, Patrick Alrassy, Pengchuan Zhang, Peng-wei Li, Petar Vasic, Peter Weng, Prajjwal Bhargava, Pratik Dubal, Praveen Krishnan, Punit Singh Koura,Puxin Xu, Qing He, Qingxiao Dong, Ragavan Srinivasan, Raj Ganapathy, Ramon Calderer, Ricardo Sil-veira Cabral, Robert Stojnic, Roberta Raileanu, Rohan Maheswari, Rohit Girdhar, Rohit Patel, RomainSauvestre, Ronnie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hos-seini, Sahana Chennabasappa, Sanjay Singh, Sean Bell, Seohyun Sonia Kim, Sergey Edunov, ShaoliangNie, Sharan Narang, Sharath Raparthy, Sheng Shen, Shengye Wan, Shruti Bhosale, Shun Zhang, SimonVandenhende, Soumya Batra, Spencer Whitman, Sten Sootla, Stephane Collot, Suchin Gururangan, Syd-ney Borodinsky, Tamar Herman, Tara Fowler, Tarek Sheasha, Thomas Georgiou, Thomas Scialom, TobiasSpeckbacher, Todor Mihaylov, Tong Xiao, Ujjwal Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ra-manathan, Viktor Kerkez, Vincent Gonguet, Virginie Do, Vish Vogeti, Vítor Albiero, Vladan Petrovic,Weiwei Chu, Wenhan Xiong, Wenyin Fu, Whitney Meers, Xavier Martinet, Xiaodong Wang, XiaofangWang, Xiaoqing Ellen Tan, Xide Xia, Xinfeng Xie, Xuchao Jia, Xuewei Wang, Yaelle Goldschlag, YasheshGaur, Yasmine Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao, Zacharie DelpierreCoudert, Zheng Yan, Zhengxing Chen, Zoe Papakipos, Aaditya Singh, Aayushi Srivastava, Abha Jain,Adam Kelsey, Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria, Ahuva Goldstand, Ajay Menon, AjaySharma, Alex Boesenberg, Alexei Baevski, Allie Feinstein, Amanda Kallet, Amit Sangani, Amos Teo,Anam Yunus, Andrei Lupu, Andres Alvarado, Andrew Caples, Andrew Gu, Andrew Ho, Andrew Poul-ton, Andrew Ryan, Ankit Ramchandani, Annie Dong, Annie Franco, Anuj Goyal, Aparajita Saraf, Arka-bandhu Chowdhury, Ashley Gabriel, Ashwin Bharambe, Assaf Eisenman, Azadeh Yazdan, Beau James,Ben Maurer, Benjamin Leonhardi, Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi Paranjape, BingLiu, Bo Wu, Boyu Ni, Braden Hancock, Bram Wasti, Brandon Spence, Brani Stojkovic, Brian Gamido,Britt Montalvo, Carl Parker, Carly Burton, Catalina Mejia, Ce Liu, Changhan Wang, Changkyu Kim,Chao Zhou, Chester Hu, Ching-Hsiang Chu, Chris Cai, Chris Tindal, Christoph Feichtenhofer, CynthiaGao, Damon Civin, Dana Beaty, Daniel Kreymer, Daniel Li, David Adkins, David Xu, Davide Testuggine,Delia David, Devi Parikh, Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le, Dustin Holland, Ed-ward Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily Hahn, Emily Wood, Eric-TuanLe, Erik Brinkman, Esteban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun, Felix Kreuk, Feng Tian,Filippos Kokkinos, Firat Ozgenel, Francesco Caggioni, Frank Kanayet, Frank Seide, Gabriela Medina Flo-rez, Gabriella Schwarz, Gada Badeer, Georgia Swee, Gil Halpern, Grant Herman, Grigory Sizov, Guangyi,Zhang, Guna Lakshminarayanan, Hakan Inan, Hamid Shojanazeri, Han Zou, Hannah Wang, HanwenZha, Haroun Habeeb, Harrison Rudolph, Helen Suk, Henry Aspegren, Hunter Goldman, Hongyuan Zhan,Ibrahim Damlaj, Igor Molybog, Igor Tufanov, Ilias Leontiadis, Irina-Elena Veliche, Itai Gat, Jake Weiss-man, James Geboski, James Kohli, Janice Lam, Japhet Asher, Jean-Baptiste Gaya, Jeff Marcus, Jeff Tang,Jennifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy Teboul, Jessica Zhong, Jian Jin, Jingyi Yang, JoeCummings, Jon Carvill, Jon Shepard, Jonathan McPhie, Jonathan Torres, Josh Ginsburg, Junjie Wang,Kai Wu, Kam Hou U, Karan Saxena, Kartikay Khandelwal, Katayoun Zand, Kathy Matosich, KaushikVeeraraghavan, Kelly Michelena, Keqian Li, Kiran Jagadeesh, Kun Huang, Kunal Chawla, Kyle Huang,Lailin Chen, Lakshya Garg, Lavender A, Leandro Silva, Lee Bell, Lei Zhang, Liangpeng Guo, Licheng Yu,Liron Moshkovich, Luca Wehrstedt, Madian Khabsa, Manav Avalani, Manish Bhatt, Martynas Mankus,Matan Hasson, Matthew Lennie, Matthias Reso, Maxim Groshev, Maxim Naumov, Maya Lathi, MeghanKeneally, Miao Liu, Michael L. Seltzer, Michal Valko, Michelle Restrepo, Mihir Patel, Mik Vyatskov,Mikayel Samvelyan, Mike Clark, Mike Macey, Mike Wang, Miquel Jubert Hermoso, Mo Metanat, Moham-

mad Rastegari, Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White, Navyata Bawa,Nayan Singhal, Nick Egebo, Nicolas Usunier, Nikhil Mehta, Nikolay Pavlovich Laptev, Ning Dong, Nor-man Cheng, Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin Kent, Parth Parekh,Paul Saab, Pavan Balaji, Pedro Rittner, Philip Bontrager, Pierre Roux, Piotr Dollar, Polina Zvyag-ina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel Rodriguez, Rafi Ayub,Raghotham Murthy, Raghu Nayani, Rahul Mitra, Rangaprabhu Parthasarathy, Raymond Li, RebekkahHogan, Robin Battey, Rocky Wang, Russ Howes, Ruty Rinott, Sachin Mehta, Sachin Siby, Sai JayeshBondu, Samyak Datta, Sara Chugh, Sara Hunt, Sargun Dhillon, Sasha Sidorov, Satadru Pan, SaurabhMahajan, Saurabh Verma, Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lindsay, Shaun Lindsay, ShengFeng, Shenghao Lin, Shengxin Cindy Zha, Shishir Patil, Shiva Shankar, Shuqiang Zhang, Shuqiang Zhang,Sinong Wang, Sneha Agarwal, Soji Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen Chen, SteveKehoe, Steve Satterfield, Sudarshan Govindaprasad, Sumit Gupta, Summer Deng, Sungmin Cho, SunnyVirk, Suraj Subramanian, Sy Choudhury, Sydney Goldman, Tal Remez, Tamar Glaser, Tamara Best, ThiloKoehler, Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim Matthews, Timothy Chou, Tzook Shaked,Varun Vontimitta, Victoria Ajayi, Victoria Montanez, Vijai Mohan, Vinay Satish Kumar, Vishal Mangla,Vlad Ionescu, Vlad Poenaru, Vlad Tiberiu Mihailescu, Vladimir Ivanov, Wei Li, Wenchen Wang, WenwenJiang, Wes Bouaziz, Will Constable, Xiaocheng Tang, Xiaojian Wu, Xiaolan Wang, Xilun Wu, XinboGao, Yaniv Kleinman, Yanjun Chen, Ye Hu, Ye Jia, Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi,Youngjin Nam, Yu, Wang, Yu Zhao, Yuchen Hao, Yundi Qian, Yunlu Li, Yuzi He, Zach Rait, ZacharyDeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang, Zhiwei Zhao, and Zhiyu Ma. The llama 3 herd ofmodels, 2024. URL https://arxiv.org/abs/2407.21783.



An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu,Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang,Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue,Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, YangFan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5technical report. arXiv preprint arXiv:2412.15115, 2024.





Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts,Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi,Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, VinodkumarPrabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, MichaelIsard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, HenrykMichalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito,David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, ShivaniAgrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, AitorLewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang,Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, DouglasEck, Jeff Dean, Slav Petrov, and Noah Fiedel. PaLM: Scaling language modeling with pathways, 2022.arXiv:2204.02311.





Dan Hendrycks and Kevin Gimpel. Bridging nonlinearities and stochastic regularizers with gaussian errorlinear units, 2016. arXiv:1606.08415.





Stefan Elfwing, Eiji Uchibe, and Kenji Doya. Sigmoid-weighted linear units for neural network functionapproximation in reinforcement learning, 2017. URL https://arxiv.org/abs/1702.03118.





Yann N. Dauphin, Angela Fan, Michael Auli, and David Grangier. Language modeling with gated convolu-tional networks, 2017. URL https://arxiv.org/abs/1612.08083.





Noam Shazeer. GLU variants improve transformer, 2020. arXiv:2002.05202.





Jianlin Su, Yu Lu, Shengfeng Pan, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotaryposition embedding, 2021.





Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Proc. of ICLR, 2015.





Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In Proc. of ICLR, 2019.





Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Nee-lakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, GretchenKrueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter,Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark,Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language modelsare few-shot learners. In Proc. of NeurIPS, 2020.





Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, ScottGray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models, 2020.arXiv:2001.08361.





Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford,Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland,Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan,Erich Elsen, Jack W. Rae, Oriol Vinyals, and Laurent Sifre. Training compute-optimal large languagemodels, 2022. arXiv:2203.15556.





Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The curious case of neural text degenera-tion. In Proc. of ICLR, 2020.





Yao-Hung Hubert Tsai, Shaojie Bai, Makoto Yamada, Louis-Philippe Morency, and Ruslan Salakhutdinov.Transformer dissection: An unified understanding for transformer‘s attention via the lens of kernel. InKentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan, editors, Proceedings of the 2019 Conferenceon Empirical Methods in Natural Language Processing and the 9th International Joint Conference onNatural Language Processing (EMNLP-IJCNLP), pages 4344–4353, Hong Kong, China, November 2019.Association for Computational Linguistics. doi: 10.18653/v1/D19-1443. URL https://aclanthology.org/D19-1443/.





Amirhossein Kazemnejad, Inkit Padhi, Karthikeyan Natesan, Payel Das, and Siva Reddy. The impact ofpositional encoding on length generalization in transformers. In Thirty-seventh Conference on NeuralInformation Processing Systems, 2023. URL https://openreview.net/forum?id=Drrl2gcjzl.

