package net.liaocy.ml4j.lda;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 *
 * @author liaocy
 */
public class Test {
    public static void main(String[]  args){
        
        //Load each word from each doument
        List<Token> tokens = new ArrayList<>();
        tokens.add(new Token(5, 1));
        tokens.add(new Token(27, 1));
        tokens.add(new Token(27, 2));
        tokens.add(new Token(1752, 2));
        tokens.add(new Token(27, 2));
        //.....
        
        LDA lda = new LDA(100, tokens);
        
        for(int k = 0; k < 1000; k++){
            lda.GibbsSampling();
        }
    
        //Average distance of overlaps in each topic distribution (the smaller the better)
        TopicDensity td = new TopicDensity(lda);
        double avgdis = td.AveDis();
        double perplexity1 = td.Perplexity1();
        double perplexity2 = td.Perplexity2();
        
        
        int docid, topicid, wordid;
        double topicprob, wordprob;
        Map.Entry<Integer, Double> mapTopic, mapWord;
        List<Map.Entry<Integer, Double>> listTopics, listWords;
        
        //output theta (documents : topics: probabilities)
        Map<Integer,Map<Integer,Double>> mapTheta = lda.getTheta();
        for (Map.Entry<Integer, Map<Integer, Double>> theta : mapTheta.entrySet()) {
            listTopics = new ArrayList<>(theta.getValue().entrySet());
            //sort by probability descend
            Collections.sort(listTopics, (Map.Entry<Integer, Double> o1, Map.Entry<Integer, Double> o2) -> {
                return -Double.compare(o1.getValue(), o2.getValue());
            });
            docid = theta.getKey();
            System.out.print(docid + ",");
            for (int i = 0; i < 10; i++) {
                mapTopic = listTopics.get(i);
                topicid = mapTopic.getKey();
                topicprob = mapTopic.getValue();
                System.out.print(topicid + "(" + topicprob + ")" + ",");
            }
            System.out.println("");
        }
        
        //output phi (topics : words : probabilities)
        Map<Integer,Map<Integer,Double>> mapPhi = lda.getPhi();
        for (Map.Entry<Integer, Map<Integer, Double>> phi : mapPhi.entrySet()) {
            listWords = new ArrayList<>(phi.getValue().entrySet());
            //sort by probability descend
            Collections.sort(listWords, (Map.Entry<Integer, Double> o1, Map.Entry<Integer, Double> o2) -> {
                return -Double.compare(o1.getValue(), o2.getValue());
            });
            topicid = phi.getKey();
            System.out.print(topicid + ",");
            for (int i = 0; i < 10; i++) {
                mapWord = listWords.get(i);
                wordid = mapWord.getKey();
                wordprob = mapWord.getValue();
                System.out.print(wordid + "(" + wordprob + ")" + ",");
            }
            System.out.println("");
        }
    }
}
