package tokenizer;

import java.io.IOException;

public class EncodeFiles {
    public static void main(String[] args) throws IOException {
        Tokenizer myTokenizer = new Tokenizer("src/saved_tokenizers/main/vocab.txt");

        for (int i = 0; i <= 2; i++) {
            myTokenizer.encodeFile(String.format("data/code/%04d.txt", i), String.format("data/outputs/code/%04d.npy", i));
        }

        for (int i = 0; i <= 445; i++) {
            myTokenizer.encodeFile(String.format("data/fineweb/%04d.txt", i), String.format("data/outputs/fineweb/%04d.npy", i));
        }
    }
}
