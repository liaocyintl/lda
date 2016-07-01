package net.liaocy.ml4j.lda;


import java.util.*;

/**
 *
 * @author Liaocy
 */

public class LDA {

    int D; // number of document
    int T; // number of topic
    int V; // number of unique word (vocabulary)
    int wordCount[][];
    int docCount[][];
    int topicCount[];
    // hyper parameter
    double alpha, beta;
    Token tokens[];
    double P[];
    // topic assignment
    int z[];
    Random rand;
    
    List<Integer> docs;
    List<Integer> words;

    public LDA(int topicNum, List<Token> tlist) {
        int documentNum = 0, vocabularyNum = 0;
        
        docs = new ArrayList<>();
        words = new ArrayList<>();
        List<Token> newtlist = new ArrayList<>();
        
        for(Token token : tlist){
            if(!docs.contains(token.docId)){
                documentNum++;
                docs.add(token.docId);
            }
            if(!words.contains(token.wordId)){
                vocabularyNum++;
                words.add(token.wordId);
            }
            newtlist.add(new Token(docs.indexOf(token.docId), words.indexOf(token.wordId)));
        }
        
        this.init(documentNum, topicNum, vocabularyNum, newtlist, 50.0 / topicNum, 0.1, 666);
    }

    private void init(int documentNum, int topicNum, int vocabularyNum, List<Token> tlist,
            double alpha, double beta, int seed) {
        D = documentNum;
        T = topicNum;
        V = vocabularyNum;

        wordCount = new int[V][T];
        topicCount = new int[T];
        docCount = new int[D][T];
        tokens = tlist.toArray(new Token[0]);
        z = new int[tokens.length];
        this.alpha = alpha;
        this.beta = beta;
        P = new double[T + 1];
        rand = new Random(seed);
        
        for (int i = 0; i < z.length; ++i) {
            Token t = tokens[i];
            int assign = rand.nextInt(T);
            wordCount[t.wordId][assign]++;
            docCount[t.docId][assign]++;
            topicCount[assign]++;
            z[i] = assign;
        }
    }

    public void GibbsSampling() {
        for (int i = 0; i < z.length; ++i) {
            Token token = tokens[i];
            int assign = z[i];
            //take out old topic
            {
                wordCount[token.wordId][assign]--;
                docCount[token.docId][assign]--;
                topicCount[assign]--;
            }
            //resampling
            {
                for (int t = 0; t < T; ++t) {
                    P[t + 1] = P[t] + (docCount[token.docId][t] + alpha)
                            * (wordCount[token.wordId][t] + beta)
                            / (topicCount[t] + V * beta);
                }
                double u = rand.nextDouble() * P[T];
                for (int t = 1; t <= T; ++t) {
                    if (u < P[t]) {
                        assign = t - 1;
                        break;
                    }
                }
            }
            //assign new topic
            {
                wordCount[token.wordId][assign]++;
                docCount[token.docId][assign]++;
                topicCount[assign]++;
                z[i] = assign;
            }
        }
    }

    public Map<Integer,Map<Integer,Double>> getTheta() {
        double theta[][] = new double[D][T];
        for (int i = 0; i < D; ++i) {
            double sum = 0.0;
            for (int j = 0; j < T; ++j) {
                theta[i][j] = alpha + docCount[i][j];
                sum += theta[i][j];
            }
            // normalize
            double sinv = 1.0 / sum;
            for (int j = 0; j < T; ++j) {
                theta[i][j] *= sinv;
            }
        }
        Map<Integer,Map<Integer,Double>> mapTheta = new HashMap<>();
        Map<Integer,Double> mapTopics;
        for (int i = 0; i < D; ++i) {
            mapTopics = new HashMap<>();
            for (int j = 0; j < T; ++j) {
                mapTopics.put(j, theta[i][j]);
            }
            mapTheta.put(docs.get(i), mapTopics);
        }
        return mapTheta;
    }

    public Map<Integer,Map<Integer,Double>> getPhi() {
        double phi[][] = new double[T][V];
        for (int i = 0; i < T; ++i) {
            double sum = 0.0;
            for (int j = 0; j < V; ++j) {
                phi[i][j] = beta + wordCount[j][i];
                sum += phi[i][j];
            }
            // normalize
            double sinv = 1.0 / sum;
            for (int j = 0; j < V; ++j) {
                phi[i][j] *= sinv;
            }
        }
        Map<Integer,Map<Integer,Double>> mapPhi = new HashMap<>();
        Map<Integer,Double> mapWords;
        for (int i = 0; i < T; ++i) {
            mapWords = new HashMap<>();
            for (int j = 0; j < V; ++j) {
                mapWords.put(words.get(j), phi[i][j]);
            }
            mapPhi.put(i, mapWords);
        }
        return mapPhi;
    }
}
