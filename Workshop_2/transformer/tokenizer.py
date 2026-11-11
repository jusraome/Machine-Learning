import json
from pathlib import Path


class Tokenizer:
    
    def __init__(self, vocab_path=None):
        if vocab_path is None:
            vocab_path = Path(__file__).resolve().parents[2] / "Workshop_1" / "data" / "vocab.json"

        with open(vocab_path, encoding="utf-8") as file:
            vocab = json.load(file)

        self.charToIdx = vocab["stoi"]
        self.idxToChar = {int(index): char for char, index in self.charToIdx.items()}
        self.unk_token = vocab["unk_token"]
        self.unk_id = vocab["unk_index"]

    def encode(self, text):
        return [self.charToIdx.get(char, self.unk_id) for char in text]

    def decode(self, ids):
        return "".join(
            self.idxToChar.get(int(index), self.unk_token)
            for index in ids
            if int(index) != self.unk_id
        )

    def vocab_size(self):
        return len(self.charToIdx)
