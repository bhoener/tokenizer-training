class Tokenizer:
    """Python class for the tokenizer, only to be used for inference"""

    def __init__(self, vocab_file: str, stoi_file: str):
        self.vocab = self.__read_set(vocab_file)
        self.stoi = self.__read_dict(stoi_file)
        self.itos = {v: k for k, v in self.stoi.items()}

    def __read_set(self, src_file: str) -> set[str]:
        with open(src_file, "r", encoding="utf-8") as f:
            return set(
                word[1:] if len(word) > 0 and word[0] == " " else word
                for word in f.read()[1:-1].split(",")
            )

    def __read_dict(self, src_file: str) -> dict:
        with open(src_file, "r", encoding="utf-8") as f:
            out = {}
            for pair in f.read()[1:-1].split(","):
                if pair.find("=") != -1:
                    if pair[0] == " ":
                        pair = pair[1:]
                    kv_split = pair.split("=")

                    k, v = "".join(kv_split[:-1]), kv_split[-1]
                    v = int(v)

                    out[k] = v
        return out

    def encode(self, src: str) -> list[int]:
        tokens = []
        buffer = str(src[0])

        for c in src[1:]:
            if buffer in self.vocab and buffer + c not in self.vocab:
                tokens.append(self.stoi[buffer])
                buffer = ""

            buffer += c

        while len(buffer) > 0:
            token_buffer = str(buffer)

            while token_buffer not in self.vocab:
                token_buffer = token_buffer[:-1]

            tokens.append(self.stoi[token_buffer])
            buffer = buffer[len(token_buffer) :]
        return tokens

    def decode(self, src: list[int]) -> str:
        return "".join(self.itos[token] for token in src)


def main() -> None:
    filepath = "src/saved_tokenizers/shakespeare/"
    enc = Tokenizer(filepath + "vocab.txt", filepath + "stoi.txt")

    print(enc.encode("ROMEO: Hello world"))
    for token in enc.encode("ROMEO: hello world"):
        print(f"{token}=|{enc.decode([token])}|")
    print(enc.decode(enc.encode("ROMEO: Hello world")))


if __name__ == "__main__":
    main()
