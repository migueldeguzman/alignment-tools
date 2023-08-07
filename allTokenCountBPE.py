from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from collections import Counter
import os

def train_bpe_tokenizer(file_path):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

    files = [file_path]
    tokenizer.train(files, trainer)

    return tokenizer

def count_all_token_occurrences(file_path):
    tokenizer = train_bpe_tokenizer(file_path)
    token_counts = Counter()

    with open(file_path, 'r') as file:
        for line in file:
            encoding = tokenizer.encode(line)
            token_counts.update(encoding.tokens)

    return token_counts

counts = count_all_token_occurrences('/Users/migueldeguzman/Desktop/modFDTGPT2xl/ai.text')

# Show the top 10 most common tokens
for token, count in counts.most_common(1000000):
    print(f"The token '{token}' occurs {count} times in the dataset.")
