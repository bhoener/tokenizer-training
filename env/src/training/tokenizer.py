class Tokenizer:
    """Python class for the tokenizer, only to be used for inference"""

    def __init__(self, vocab_file: str):
        self.vocab = self.__read_dict(vocab_file)
        self.stoi = {v: k for k, v in self.vocab.items()}

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

    def encode(self, src: str) -> list[int]:
        tokens = src.encode("utf-8")

        while True:
            merges = []

            for prev, current in zip(tokens, tokens[1:]):
                merges.append((prev, current))

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
    
    filepath = "src/saved_tokenizers/main/"
    enc = Tokenizer(filepath + "vocab.txt")
    
    dl = DataLoader("data/outputs/fineweb/", B=4, T=32)
    
    xs, ys = dl.next()

    print("|", end="")
    for token in enc.encode("Romeo.\nhello world\n\nMERCUTIO. Sigma Fortnite balls"):
        print(f"{enc.decode([token])}|", end="")
        
    for token in xs.view(-1).cpu().numpy():
        print(f"{enc.decode([token])}|", end="")
        
    for token in xs.view(-1).cpu().numpy():
        print(f"{enc.decode([token])}", end="")


if __name__ == "__main__":
    main()
