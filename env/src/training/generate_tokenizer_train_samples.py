
FILE_SIZE = 5_000_000

start_dir = "data/"
input_dirs = ["code/", "fineweb/"]
out_dir = "data/tokenizer_train/"


for dir in input_dirs:
    with open(start_dir + dir + "0000.txt", "r", encoding="utf-8") as f:
        with open(out_dir + dir + "0000.txt", "w+", encoding="utf-8") as out:
            out.write(f.read()[:FILE_SIZE])
