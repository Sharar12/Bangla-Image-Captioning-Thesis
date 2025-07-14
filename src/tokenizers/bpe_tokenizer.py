from transformers import AutoTokenizer

class BanglaBpeTokenizer:
    def __init__(self, model_name: str = "sagorsarker/banglabert"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, text: str):
        return self.tokenizer.encode(text, add_special_tokens=True)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    @property
    def vocab_size(self):
        return len(self.tokenizer)
