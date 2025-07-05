from collections import Counter
from typing import List, Dict

class Vocabulary:
    def __init__(self, freq_threshold: int = 3):
        self.freq_threshold = freq_threshold
        self.itos: Dict[int, str] = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi: Dict[str, int] = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list: List[str]):
        frequencies = Counter()
        idx = 4
        for sentence in sentence_list:
            for word in sentence.split():
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text: str) -> List[int]:
        tokenized_text = text.split()
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text]
