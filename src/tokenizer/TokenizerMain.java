package tokenizer;
import java.util.*;
import java.io.IOException;

public class TokenizerMain {
    public static void main(String[] args) throws IOException, InvalidTokenException {
        Tokenizer myTokenizer = new Tokenizer(10000, 7500);

        myTokenizer.train(new String[] {"src/data/shakespeare.txt"});

        String inText = "MONTAGUE.\r\n" + //
                        "Both by myself and many other friends;\r\n" + //
                        "But he, his own affections' counsellor,\r\n" + //
                        "Is to himself—I will not say how true-\r\n" + //
                        "But to himself so secret and so close,\r\n" + //
                        "So far from sounding and discovery,\r\n" + //
                        "As is the bud bit with an envious worm\r\n" + //
                        "Ere he can spread his sweet leaves to the air,\r\n" + //
                        "Or dedicate his beauty to the sun.\r\n" + //
                        "Could we but learn from whence his sorrows grow,\r\n" + //
                        "We would as willingly give cure as know.";

        System.out.println(myTokenizer.encode(inText));
        List<Integer> tokens = myTokenizer.encode(inText);
        for (Integer token: tokens) {
            if (token != null) {
                System.out.print(myTokenizer.decodeSingle((int) token) + "|");
            } else {
                System.out.print("NULL TOKEN!!!!!!!");
            }
        }

        String saveDir = "src/saved_tokenizers/shakespeare/";
        myTokenizer.saveState(saveDir + "vocab.txt");

        myTokenizer.encodeFile("src/data/testing/input.txt", "test_out.npy");
    }
}
