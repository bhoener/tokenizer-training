package tokenizer;

import java.io.IOException;

public class TokenizerMain {
    public static void main(String[] args) throws IOException {
        Tokenizer myTokenizer = new Tokenizer(500);

        myTokenizer.train(new String[] {"src/data/shakespeare.txt"});
    }
}
