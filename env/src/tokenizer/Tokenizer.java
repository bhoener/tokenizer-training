package tokenizer;

import java.util.*;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.regex.*;

public class Tokenizer {
    private ReversibleMap<Integer, TokenPair> vocab;
    private int vocabSize;
    private final int MAX_VOCAB_SIZE;
    private int allowMultiWord;
    private Map<String, Integer> tokenCache;
    private Pattern pretokRegex = Pattern.compile("\\s+");

    public Tokenizer(int vocabSize) {
        this.MAX_VOCAB_SIZE = vocabSize;
        this.vocabSize = 255;
        this.vocab = new ReversibleMap<>();
        this.tokenCache = new HashMap<>();
        this.buildCache();
    }

    public Tokenizer(String vocabFile) throws IOException {
        this.vocab = new ReversibleMap<>(this.loadVocab(vocabFile));
        this.vocabSize = this.vocab.size();
        this.MAX_VOCAB_SIZE = vocabSize;
        this.tokenCache = new HashMap<>();
    }

    private void buildCache() {
        for (int i: this.vocab.keySet()) {
            this.tokenCache.put(this.toString(this.vocab.getValue(i)), i);
        }
    }

    public void train(String[] files) throws IOException {
        if (MAX_VOCAB_SIZE < 1 || files.length < 1)
            throw new IllegalArgumentException();

        List<Integer> tokens = new ArrayList<>();
        for (String file : files) {
            tokens.addAll(byteTokenizeFile(file));
        }

        while (this.vocabSize < MAX_VOCAB_SIZE) {
            HashMap<TokenPair, Integer> frequencyMap = new HashMap<>();

            Iterator<Integer> tokensIter = tokens.iterator();

            int last = tokensIter.next();

            while (tokensIter.hasNext()) {
                int current = tokensIter.next();

                TokenPair combined = new TokenPair(last, current);

                Integer currentFreq = frequencyMap.get(combined); // i hate java so much this doesn't work with int
                frequencyMap.put(combined, currentFreq != null ? currentFreq + 1 : 1);

                last = current;
            }

            TokenPair maxPair = null;
            int maxFreq = 0;

            if (this.vocabSize == this.allowMultiWord)
                System.out.println("Allowing multi-word tokens");

            for (TokenPair entry : frequencyMap.keySet()) {
                int value = frequencyMap.get(entry);

                if (value > maxFreq && !this.vocab.containsValue(entry)
                        && !isMultiWord(entry) && !containsEOT(entry)) {
                    maxFreq = value;
                    maxPair = entry;
                }
            }

            this.vocab.putKV(++this.vocabSize, maxPair);

            Iterator<Integer> iter = tokens.iterator();
            ArrayList<Integer> newTokens = new ArrayList<>(tokens.size());
            while (iter.hasNext()) {
                last = iter.next();
                if (iter.hasNext()) {
                    int current = iter.next();
                    TokenPair combined = new TokenPair(last, current);

                    if (combined.equals(maxPair))
                        newTokens.add(this.vocabSize);
                    else {
                        newTokens.add(last);
                        newTokens.add(current);
                    }
                } else {
                    newTokens.add(last);
                }
            }

            tokens = newTokens;

            if (this.vocabSize % 10 == 0)
                System.out.println("New merge: " + toString(maxPair));
        }
    }

    private boolean isMultiWord(TokenPair input) {
        String inString = toString(input);
        boolean hasAlphabetic = false;
        for (int i = 0; i < inString.length(); i++) {
            char current = inString.charAt(i);
            if (Character.isAlphabetic(current))
                hasAlphabetic = true;
            if (current == ' ' && hasAlphabetic)
                return true;
        }
        return false;
    }

    private static boolean containsEOT(TokenPair input) {
        return (input.__t1() == 3 || input.__t2() == 3);
    }

    private String toString(TokenPair input) {
        boolean t1merge = this.vocab.containsKey(input.__t1());
        boolean t2merge = this.vocab.containsKey(input.__t2());
        if (!t1merge && !t2merge) {
            return String.valueOf((char) input.__t1()) + String.valueOf((char) input.__t2());
        } else if (!t1merge) {
            return String.valueOf((char) input.__t1()) + toString(this.vocab.getValue(input.__t2()));
        } else if (!t2merge) {
            return toString(this.vocab.getValue(input.__t1())) + String.valueOf((char) input.__t2());
        } else {
            return toString(this.vocab.getValue(input.__t1())) + toString(this.vocab.getValue(input.__t2()));
        }
    }

    public ArrayList<Integer> byteTokenizeFile(String filename) throws IOException {
        BufferedInputStream input = new BufferedInputStream(new FileInputStream(filename));

        ArrayList<Integer> tokens = new ArrayList<>(input.available());

        int res;

        while ((res = input.read()) != -1)
            tokens.add(res);

        input.close();

        return tokens;
    }

    public List<Integer> encode(ByteBuffer input) {
        // long t0 = System.nanoTime();
        List<Integer> tokens = new ArrayList<>(input.remaining());
        while (input.hasRemaining()) {
            tokens.add((int) (input.get() & 0xff));
        }
        // System.out.println("Adding token bytes: " + (System.nanoTime() - t0));

        while (true) {
            // t0 = System.nanoTime();
            Set<TokenPair> merges = new HashSet<>(tokens.size() * 2);
            for (int i = 1; i < tokens.size(); i++) {
                TokenPair merge = new TokenPair(tokens.get(i - 1), tokens.get(i));
                if (this.vocab.containsValue(merge))
                     merges.add(merge);
            }
            // System.out.println("Adding merges: " + (System.nanoTime() - t0));

            // t0 = System.nanoTime();
            Iterator<TokenPair> iter = merges.iterator();
            TokenPair minMerge = null;
            int minIdx = this.vocabSize;
            while (iter.hasNext()) {
                TokenPair merge = iter.next();

                if (this.vocab.containsValue(merge)) {
                    int idx = this.vocab.getKey(merge);

                    if (idx > 255) {
                        if (idx < minIdx) {
                            minMerge = merge;
                            minIdx = idx;
                        }
                    }

                }
            }

            // System.out.println("Finding min: " + (System.nanoTime() - t0));

            if (minMerge == null)
                break;


            // t0 = System.nanoTime();
            ArrayList<Integer> newTokens = new ArrayList<>(tokens.size());
            Iterator<Integer> tokensIter = tokens.iterator();

            int last = tokensIter.next();
            while (tokensIter.hasNext()) {
                int current = tokensIter.next();

                TokenPair combined = new TokenPair(last, current);

                if (combined.equals(minMerge)) {
                    newTokens.add(this.vocab.getKey(combined));
                    if (tokensIter.hasNext())
                        current = tokensIter.next();
                } else {
                    newTokens.add(last);
                    if (!tokensIter.hasNext())
                        newTokens.add(current);
                }

                last = current;
            }

            tokens = newTokens;
            // System.out.println("Replacing tokens: " + (System.nanoTime() - t0));
        }

        return tokens;
    }

    public List<Integer> encode(String input) {
        String[] pretokens = this.pretokRegex.split(input);

        List<Integer> tokens = new ArrayList<>(pretokens.length * 2);
        
        for (String pretoken: pretokens) {
            if (this.tokenCache.containsKey(pretoken)) {
                tokens.add(this.tokenCache.get(pretoken));
            } else {
                tokens.addAll(this.encode(StandardCharsets.UTF_8.encode(pretoken)));
            }
        }
        return tokens;
    }

    public void encodeFile(String inputFile, String outputFile, int chunkSize) throws IOException {
        byte[] buf = new byte[chunkSize];
        BufferedInputStream input = new BufferedInputStream(new FileInputStream(inputFile));
        DataOutputStream out = new DataOutputStream(new FileOutputStream(outputFile));
        while (input.read(buf) != -1) {
            long t0 = System.nanoTime();
            List<Integer> tokens = this.encode(new String(buf, StandardCharsets.UTF_8));
            long time = System.nanoTime() - t0;
            System.out.println("Tok/s: " + (tokens.size() / (time / 1e9)));
            for (int t : tokens) {
                out.writeInt(t);
            }
        }
        input.close();
        out.close();
    }

    public void encodeFile(String inputFile, String outputFile) throws IOException {
        this.encodeFile(inputFile, outputFile, 1024);
    }

    public String decode(List<Integer> input) throws InvalidTokenException {
        String output = "";
        for (int token : input) {
            output += this.decodeSingle(token);
        }
        return output;
    }

    public String decodeSingle(int input) throws InvalidTokenException {
        if (!this.vocab.containsKey(input) && (input < 0 || input > this.vocabSize))
            throw new InvalidTokenException(Integer.toString(input));
        return input > 255 ? toString(this.vocab.getValue(input)) : String.valueOf((char) input);
    }

    public void saveState(String vocabFile) throws IOException {
        PrintStream vocabOut = new PrintStream(new File(vocabFile));

        for (int k : this.vocab.keySet()) {
            vocabOut.println(k + "=" + this.vocab.getValue(k).toString());
        }

        vocabOut.close();
    }

    class TokenPair {
        private int t1;
        private int t2;

        public TokenPair(int t1, int t2) {
            this.t1 = t1;
            this.t2 = t2;
        }

        @Override
        public boolean equals(Object other) {
            if (!(other instanceof TokenPair))
                return false;
            return this.t1 == ((TokenPair) other).__t1() && this.t2 == ((TokenPair) other).__t2();
        }

        @Override
        public int hashCode() {
            return this.t1 * 31 + this.t2 * 17;
        }

        @Override
        public String toString() {
            return String.format("(%d,%d)", this.t1, this.t2);
        }

        public int __t1() {
            return this.t1;
        }

        public int __t2() {
            return this.t2;
        }
    }

    private HashMap<Integer, TokenPair> loadVocab(String filename) throws IOException {
        Scanner input = new Scanner(new File(filename));

        HashMap<Integer, TokenPair> output = new HashMap<>();

        while (input.hasNextLine()) {
            String line = input.nextLine();

            int equalIdx = line.indexOf('=', 0);

            int key = Integer.parseInt(line.substring(0, equalIdx));

            String tkString = line.substring(equalIdx + 2, line.length() - 1);
            int commaIdx = tkString.indexOf(',');

            int t1 = Integer.parseInt(tkString.substring(0, commaIdx));
            int t2 = Integer.parseInt(tkString.substring(commaIdx + 1));

            TokenPair value = new TokenPair(t1, t2);

            output.put(key, value);
        }

        input.close();

        return output;
    }

    public ReversibleMap<Integer, TokenPair> __vocab() {
        return this.vocab;
    }
}
