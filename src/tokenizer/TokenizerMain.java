package tokenizer;

import java.io.IOException;

public class TokenizerMain {
    public static void main(String[] args) throws IOException {
        Tokenizer myTokenizer = new Tokenizer(500);

        myTokenizer.train(new String[] {"src/data/shakespeare.txt"});

        System.out.println(myTokenizer.encode("ROMEO: Hello world"));
        for (int token: myTokenizer.encode("ROMEO: Hello world")) {
            System.out.print(myTokenizer.decodeSingle(token) + "|");
        }

        myTokenizer.tokenizeFile("src/data/testing/input.txt", "output.npy");

        String saveDir = "src/saved_tokenizers/shakespeare/";
        myTokenizer.saveState(saveDir + "vocab.txt", saveDir + "stoi.txt");
    }
}
