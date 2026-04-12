import re


class Tokenizer:
    """Python class for the tokenizer, only to be used for inference"""

    def __init__(self, vocab_file: str):
        self.vocab = self.__read_dict(vocab_file)
        self.stoi = {v: k for k, v in self.vocab.items()}
        self.token_cache = {self.decode_single(tok): tok for tok in self.vocab.keys()}

    def __read_dict(self, src_file: str) -> dict:
        with open(src_file, "r", encoding="utf-8") as f:
            out = {}
            for pair in f.read().splitlines():
                if pair.find("=") != -1:
                    if pair[0] == " ":
                        pair = pair[1:]
                    kv_split = pair.split("=")

                    k, v = "".join(kv_split[:-1]), kv_split[-1]
                    pair = (int(v[1:-1].split(",")[0]), int(v[1:-1].split(",")[1]))

                    out[int(k)] = pair
        return out

    def __get_min(self, merges: list[tuple]):
        minimum = merges[0] if merges[0] in self.stoi else None
        for merge in merges:
            if merge in self.stoi and (
                minimum is None or self.stoi[merge] < self.stoi[minimum]
            ):
                minimum = merge
        return minimum

    def __encode_chunk(self, tokens: list[int]) -> list[int]:
        while True:
            merges = []

            for prev, current in zip(tokens, tokens[1:]):
                merges.append((prev, current))

            if len(merges) < 1:
                break

            min_merge = self.__get_min(merges)

            if min_merge is None:
                break

            tokens_new = []

            i = 1
            prev = tokens[i - 1]
            while i < len(tokens):
                current = tokens[i]

                combined = (prev, current)

                if combined == min_merge:
                    tokens_new.append(self.stoi[combined])
                    if i < len(tokens) - 1:
                        i += 1
                        current = tokens[i]
                        if i >= len(tokens) - 1:
                            tokens_new.append(current)
                else:
                    tokens_new.append(prev)
                    if i >= len(tokens) - 1:
                        tokens_new.append(current)
                prev = current

                i += 1

            tokens = tokens_new

        return tokens

    def encode(self, src: str) -> list[int]:
        pattern = r"\n[A-Za-z]+|\n|\s+[A-Za-z]+|\d|[^\w\s]"  # thanks chatgippity
        split = [sp.encode("utf-8") for sp in re.findall(pattern, src)]

        tokens = []
        for sp in split:
            if sp in self.token_cache:
                tokens.append(self.token_cache[sp])
            else:
                tokens.extend(self.__encode_chunk([int(c) for c in sp]))
        return tokens

    def decode(self, src: list[int]) -> str:
        return "".join(self.decode_single(token) for token in src)

    def decode_single(self, token: int) -> str:
        if 0 < token < 256:
            return chr(token)
        else:
            return self.decode_single(self.vocab[token][0]) + self.decode_single(
                self.vocab[token][1]
            )

    @property
    def vocab_size(self):
        return len(self.vocab) + 1


def main() -> None:
    from dataloader import DataLoader

    filepath = "src/saved_tokenizers/updated/"
    enc = Tokenizer(filepath + "vocab.txt")

    dl = DataLoader("data/outputs/fineweb/", B=4, T=32)

    xs, ys = dl.next()

    print("|", end="")
    src = """The Independent Jane
For all the love, romance and scandal in Jane Austen’s books, what they are really about is freedom and independence. Independence of thought and the freedom to choose.
Elizabeth’s refusal of Mr. Collins offer of marriage showed an independence"""
    tokens = enc.encode(src)
    for token in tokens:
        print(f"{enc.decode([token])}|", end="")
    print("Num tokens:", len(tokens))
    print("Tokens per character:", len(tokens) / len(src))


if __name__ == "__main__":
    main()
