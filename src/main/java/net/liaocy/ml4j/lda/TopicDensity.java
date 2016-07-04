/*
 *  license MIT
 */
package net.liaocy.ml4j.lda;
/**
 *
 * @author liaocy
 */
public class TopicDensity {
    LDA lda;
    double[][] theta;
    double[][] phi;
    
    /**
     *
     * @param lda
     */
    public TopicDensity(LDA lda){
        this.lda = lda;
        CaculateTheta();
        CaculatePhi();
    }
    
    private void CaculateTheta(){
        theta = new double[lda.D][lda.T];
        for (int i = 0; i < lda.D; ++i) {
            double sum = 0.0;
            for (int j = 0; j < lda.T; ++j) {
                theta[i][j] = lda.docCount[i][j] + lda.alpha;
                sum += theta[i][j];
            }
            // normalize
            double sinv = 1.0 / (sum + lda.T * lda.alpha);
            for (int j = 0; j < lda.T; ++j) {
                theta[i][j] *= sinv;
            }
        }
    }
    
    private void CaculatePhi(){
        phi = new double[lda.T][lda.V];
        for (int i = 0; i < lda.T; ++i) {
            double sum = 0.0;
            for (int j = 0; j < lda.V; ++j) {
                phi[i][j] = lda.wordCount[j][i] + lda.beta;
                sum += phi[i][j];
            }
            // normalize
            double sinv = 1.0 / (sum + lda.V * lda.beta);
            for (int j = 0; j < lda.V; ++j) {
                phi[i][j] *= sinv;
            }
        }
    }
    
    private double Corre(int t1, int t2){
        double temp = .0, temp1 = .0, temp2 = .0;
        for(int v = 0; v < this.lda.V; v++){
            temp += phi[t1][v] * phi[t2][v];
            temp1 += phi[t1][v] * phi[t1][v];
            temp2 += phi[t2][v] * phi[t2][v];
        }
        return temp / (Math.sqrt(temp1) * Math.sqrt(temp2));
    }
    
    /**
     * A parameter optimation method from https://github.com/skitaoka/nlp-fun/blob/master/floodgate/LDA.java
     * @return double: the value of perplexity
     */
    public double Perplexity1(){
        double perplexity = 0.0;
        for (Token token : lda.tokens) {
            double prob = .0;
            for(int t = 0; t < lda.T; t++){
                prob += theta[token.docId][t] * phi[t][token.wordId];
            }
            
            perplexity += Math.log(prob);
        }
        perplexity = -(perplexity / lda.tokens.length);
        return Math.exp(perplexity);
    }
    
    /**
     * A version from https://github.com/tedunderwood/LDA
     * @return double: the value of perplexity
     */
    public double Perplexity2(){
        double perplexity = 0.0;
        int i = 0;
        for (Token token : lda.tokens) {
            int word = token.wordId;
            int doc = token.docId;
            int t = lda.z[i];
            double prob = ((lda.docCount[doc][t] + 1) / (double)(lda.docSize[doc] + lda.D)) * ((lda.wordCount[word][t] + 1) / (double)(lda.topicCount[t] + lda.V));
            
            perplexity += Math.log(prob);
            
            i++;
        }
        perplexity = -(perplexity / lda.tokens.length);
        return Math.exp(perplexity);
    }
    
    public double Perplexity3(){
        double perplexity = 0.0;
        for (Token token : lda.tokens) {
            double prob = .0;
            for(int t = 0; t < lda.T; t++){
                prob += theta[token.docId][t] * phi[t][token.wordId];
            }
            
            perplexity += Math.log(prob);
        }
        perplexity = -(perplexity / lda.tokens.length);
        return Math.exp(perplexity);
    }
    
    /**
     *
     * @return
     */
    public double AveDis(){
        double temp = .0;
        for(int t1 = 0; t1 < this.lda.T; t1++){
            for(int t2 = t1 + 1; t2 < this.lda.T; t2++){
                temp += Corre(t1, t2);
            }
        }
        return temp / (this.lda.T * (this.lda.T - 1) / 2);
    }
    
    private int Density(int t, double r){
        int density = 0;
        for(int t1 = 0; t1 < this.lda.T; t1++){
            if(Corre(t, t1) <= r){
                density++;
            }
        }
        return density;
    }
    
    private int[] Densities(double r){
        int[] densities = new int[this.lda.T];
        for(int t = 0; t < this.lda.T; t++){
            densities[t] = Density(t, r);
        }
        return densities;
    }
    
    /**
     *
     * @param densities
     * @param n
     * @return
     */
    public int Cardinality(int[] densities, int n){
        int cardinality = 0;
        for(int t = 0; t < this.lda.T; t++){
            if(densities[t] < n){
                cardinality++;
            }
        }
        return cardinality;
    }
    
    /**
     *
     * @return
     */
    public double MainProcess(){
        double perplexity = 0.0;
        
        double r1 = AveDis();
        int[] densities = Densities(r1);
        int c = Cardinality(densities, 0);
        //densities
        
        //(this.lda.T - c);
        
        return perplexity;
    }
}
