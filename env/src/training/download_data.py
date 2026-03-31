from datasets import load_dataset
from tqdm import tqdm
import os

CHUNK_SIZE = 100_000_000
DATA_DIR = "data/fineweb/"
dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train")

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

buffer = ""
for example in tqdm(dataset):
    buffer += example["text"]