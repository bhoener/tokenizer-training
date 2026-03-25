package tokenizer;

import java.util.HashMap;
import java.util.HashSet;
import java.util.TreeSet;
import java.util.ArrayList;
import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Iterator;

public class Tokenizer {
    private TreeSet<String> vocab;
    private HashMap<String, Integer> stoi;
    private HashMap<Integer, String> itos;
    private int vocabSize;

    public Tokenizer(int vocabSize) {
        this.vocabSize = vocabSize;
        this.vocab = new TreeSet<>();
        this.stoi = new HashMap<>();
        this.itos = new HashMap<>();
    }

    public void train(String[] files) throws IOException {
        if (vocabSize < 1 || files.length < 1)
            throw new IllegalArgumentException();

        getBaseVocab(files);
        while (vocab.size() < vocabSize) {
            HashMap<String, Integer> frequencyMap = new HashMap<>();

            for (String file: files) {
                ArrayList<Integer> tokens = tokenizeFile(file);

                Iterator<Integer> tokensIter = tokens.iterator();

                int last = tokensIter.next();

                while (tokensIter.hasNext()) {
                    int current = tokensIter.next();

                    String combined = this.itos.get(last) + this.itos.get(current);

                    Integer currentFreq = frequencyMap.get(combined); // i hate java so much this doesn't work with int
                    frequencyMap.put(combined, currentFreq != null ? currentFreq + 1 : 1);

                    last = current;
                }
            }

            String maxString = null;
            int maxFreq = 0;

            for (String entry: frequencyMap.keySet()) {
                int value = frequencyMap.get(entry);

                if (value > maxFreq && !this.vocab.contains(entry)) {
                    maxFreq = value;
                    maxString = entry;
                }
            }

            this.vocab.add(maxString);

            System.out.printf("New merge: '%s'\n", maxString);
            this.buildMaps();


            // seems like there is a problem with my greedy tokenizing strategy
            // if we have the sequence .\r\n
            // and we merge the pair \r\n first,
            // we will get that \r\n is a token, but .\r is not
            // so when we try to tokenize, the greedy strategy stops at "." and .\r\n never gets tokenized
            // we could take a fixed size buffer and decrease the length until tokenizable
            // or we could keep track of all new additions and make sure we don't add one twice
            // eventually .\r will be added
        }
        
            

        System.out.println(vocab);
    }

    public void buildMaps() {
        int i = 0;
        for (String w: this.vocab) {
            this.stoi.put(w, i);
            this.itos.put(i, w);
            i++;
        }
    }

    public void getBaseVocab(String[] files) throws IOException {
        for (String file: files) {
            BufferedInputStream input = new BufferedInputStream(new FileInputStream(file));

            int res;

            while ((res = input.read()) != -1) {
                this.vocab.add(String.valueOf((char) res));
            }

            this.buildMaps();
            input.close();
        }
    }

    public ArrayList<Integer> tokenizeFile(String filename) throws IOException {
        BufferedInputStream input = new BufferedInputStream(new FileInputStream(filename));

        ArrayList<Integer> tokens = new ArrayList<>();

        int res;

        String buffer = "";
        while ((res = input.read()) != -1) {
            if (this.vocab.contains(buffer) && !this.vocab.contains(buffer + String.valueOf((char) res))) {
                tokens.add(this.stoi.get(buffer));
                buffer = "";
            }
            buffer += String.valueOf((char) res);
        }

        input.close();

        return tokens;
    }
}
