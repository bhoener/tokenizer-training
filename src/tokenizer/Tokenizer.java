package tokenizer;

import java.util.*;
import java.io.*;

public class Tokenizer {
    private Set<String> vocab;
    private Map<String, Integer> stoi;
    private Map<Integer, String> itos;
    private int vocabSize;

    public Tokenizer(int vocabSize, boolean deterministic) {
        this.vocabSize = vocabSize;
        this.vocab = deterministic ? new TreeSet<>() : new HashSet<>();
        for (char c = 0; c < 256; c++) {
            this.vocab.add(String.valueOf(c));
        }
        this.stoi = new HashMap<>();
        this.itos = new HashMap<>();
    }

    public Tokenizer(int vocabSize) {
        this(vocabSize, false);
    }

    public void train(String[] files) throws IOException {
        if (vocabSize < 1 || files.length < 1)
            throw new IllegalArgumentException();

        getBaseVocab(files);
        while (vocab.size() < vocabSize) {
            HashMap<String, Integer> frequencyMap = new HashMap<>();

            for (String file : files) {
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

            for (String entry : frequencyMap.keySet()) {
                int value = frequencyMap.get(entry);

                if (value > maxFreq && !this.vocab.contains(entry)) {
                    maxFreq = value;
                    maxString = entry;
                }
            }

            this.vocab.add(maxString);

            this.buildMaps();

            // seems like there is a problem with my greedy tokenizing strategy
            // if we have the sequence .\r\n
            // and we merge the pair \r\n first,
            // we will get that \r\n is a token, but .\r is not
            // so when we try to tokenize, the greedy strategy stops at "." and .\r\n never
            // gets tokenized
            // we could take a fixed size buffer and decrease the length until tokenizable
            // or we could keep track of all new additions and make sure we don't add one
            // twice
            // eventually .\r will be added
        }

        System.out.println(vocab);
    }

    public void buildMaps() {
        int i = 0;
        for (String w : this.vocab) {
            this.stoi.put(w, i);
            this.itos.put(i, w);
            i++;
        }
    }

    public void getBaseVocab(String[] files) throws IOException {
        for (String file : files) {
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

        String buffer = String.valueOf((char) input.read());
        while ((res = input.read()) != -1) {
            if (this.vocab.contains(buffer) && !this.vocab.contains(buffer + String.valueOf((char) res))) {
                tokens.add(this.stoi.get(buffer));
                buffer = "";
            }
            buffer += String.valueOf((char) res);
        }

        while (buffer.length() > 0) {
            String tokenBuffer = new String(buffer);

            while (!vocab.contains(tokenBuffer)) {
                tokenBuffer = tokenBuffer.substring(0, tokenBuffer.length() - 1);
            }

            tokens.add(this.stoi.get(tokenBuffer));
            buffer = buffer.substring(tokenBuffer.length());
        }

        input.close();

        return tokens;
    }

    public void tokenizeFile(String filename, String outputFile) throws IOException {
        ArrayList<Integer> tokens = tokenizeFile(filename);

        DataOutputStream output = new DataOutputStream(new FileOutputStream(outputFile));

        for (int t : tokens)
            output.writeInt(t);

        output.close();
    }

    public ArrayList<Integer> encode(String input) {
        ArrayList<Integer> tokens = new ArrayList<>();

        String buffer = String.valueOf(input.charAt(0));
        for (int i = 1; i < input.length(); i++) {
            char current = input.charAt(i);

            if (this.vocab.contains(buffer) && !this.vocab.contains(buffer + current)) {

                tokens.add(this.stoi.get(buffer));
                buffer = "";
            }
            buffer += String.valueOf(current);
        }

        while (buffer.length() > 0) {
            String tokenBuffer = new String(buffer);

            while (!vocab.contains(tokenBuffer)) {
                tokenBuffer = tokenBuffer.substring(0, tokenBuffer.length() - 1);
            }

            tokens.add(this.stoi.get(tokenBuffer));
            buffer = buffer.substring(tokenBuffer.length());
        }

        return tokens;
    }

    public int encodeSingle(String input) {
        return this.stoi.get(input);
    }

    public String decode(List<Integer> input) {
        String output = "";
        for (int token : input) {
            output += this.itos.get(token);
        }
        return output;
    }

    public String decodeSingle(int input) {
        return this.itos.get(input);
    }

    public void saveState(String vocabFile, String stoiFile) throws IOException {
        PrintStream vocabOut = new PrintStream(new File(vocabFile));
        PrintStream stoiOut = new PrintStream(new File(stoiFile));

        vocabOut.print(this.vocab);
        stoiOut.print(this.stoi);

        vocabOut.close();
        stoiOut.close();
    }

    public Set<String> __vocab() {
        return this.vocab;
    };

    public Map<String, Integer> __stoi() {
        return this.stoi;
    }

    public Map<Integer, String> __itos() {
        return this.itos;
    };
}
