from datasets import load_dataset
from tqdm import tqdm
import os

CHUNK_SIZE = 100_000_000
DATA_DIR = "data/code/"
dataset = load_dataset("glaiveai/glaive-code-assistant", split="train")

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

eot = chr(3)
chunks = 0
buffer = str(eot)


def prompt_format(questions: list, answers: list) -> list:
    return [
        f"USER:\n\n{question}\n\nASSISTANT:\n\n{answer}"
        for question, answer in zip(questions, answers)
    ]


def map_fn(examples: dict[list]) -> None:
    global buffer, chunks

    buffer += eot.join(prompt_format(examples["question"], examples["answer"]))
    if len(buffer) > CHUNK_SIZE:
        with open(f"{DATA_DIR}{chunks:04d}.txt", "w", encoding="utf-8") as f:
            f.write(buffer)
        chunks += 1
        buffer = str(eot)


dataset.map(map_fn, batched=True)

with open(f"{DATA_DIR}{chunks:04d}.txt", "w", encoding="utf-8") as f:
    f.write(buffer)