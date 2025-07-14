from transformers import AutoTokenizer

class BanglaGptTokenizer:
    def __init__(self, model_name: str = "flax-community/papluca-bengali-gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize_batch(self, texts, max_length=64):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
