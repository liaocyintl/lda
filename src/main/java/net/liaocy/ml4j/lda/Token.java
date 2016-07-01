package net.liaocy.ml4j.lda;

/**
 *
 * @author Liaocy
 */
public class Token {

    public int docId;
    public int wordId;

    public Token(int d, int w) {
        docId = d;
        wordId = w;
    }
}
