/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package net.liaocy.ml4j.lda;

/**
 *
 * @author liaoc_000
 */
public class TopicDensity {
    LDA lda;
    double[][] theta;
    double[][] phi;
    
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
                theta[i][j] = lda.alpha + lda.docCount[i][j];
                sum += theta[i][j];
            }
            // normalize
            double sinv = 1.0 / sum;
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
                phi[i][j] = lda.beta + lda.wordCount[j][i];
                sum += phi[i][j];
            }
            // normalize
            double sinv = 1.0 / sum;
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
    
    private double[][] DocWordProb(){
        double[][] docwordProb = new double[lda.D][lda.V];
        for (Token token : lda.tokens) {
            docwordProb[token.docId][token.wordId] += 1;
        }
        for(int d = 0; d < lda.D; d++){
            for(int v = 0; v < lda.V; v++){
                if(docwordProb[d][v] == .0){
                    continue;
                }
                for(int t = 0; t < lda.T; t++){
                    docwordProb[d][v] += theta[d][t] * phi[t][v] * docwordProb[d][v];
                }
            }
        }
        return docwordProb;
    }
    
    public double Perplexity(){
        double[][] docwordProb = DocWordProb();
        double perplexity = 0.0;
        for(int d = 0; d > lda.D; d++){
            
            for(int v = 0; v < lda.V; v++){
                if(docwordProb[d][v] == .0){
                    continue;
                }
                perplexity += Math.log(docwordProb[d][v]);
            }
        }
        perplexity = -(perplexity / lda.tokens.length);
        return Math.exp(perplexity);
    }
    
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
    
    public int Cardinality(int[] densities, int n){
        int cardinality = 0;
        for(int t = 0; t < this.lda.T; t++){
            if(densities[t] < n){
                cardinality++;
            }
        }
        return cardinality;
    }
    
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
