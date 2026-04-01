package tokenizer;

import java.util.*;
import java.io.IOException;

public class TokenizerTrain {
    public static void main(String[] args) throws IOException, InvalidTokenException {
        Tokenizer myTokenizer = new Tokenizer(25000, 20000);

        String[] files = new String[] { "data/tokenizer_train/code/0000.txt", "data/tokenizer_train/fineweb/0000.txt" };

        myTokenizer.train(files);
        myTokenizer.saveState("src/saved_tokenizers/main/vocab.txt");
    }
}
