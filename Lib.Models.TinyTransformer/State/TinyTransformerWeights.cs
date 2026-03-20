using Lib.Models.TinyTransformer.Configuration;

namespace Lib.Models.TinyTransformer.State;

public class TinyTransformerWeights
{
    public static Random rnd = new Random(42);
    
    public double[][] wQ = GenerateMatrix(TinyTransformerConfig.EmbeddingSize, TinyTransformerConfig.EmbeddingSize);
    public double[][] wK = GenerateMatrix(TinyTransformerConfig.EmbeddingSize, TinyTransformerConfig.EmbeddingSize);
    public double[][] wV = GenerateMatrix(TinyTransformerConfig.EmbeddingSize, TinyTransformerConfig.EmbeddingSize);
    public double[][] wO = GenerateMatrix(TinyTransformerConfig.EmbeddingSize, TinyTransformerConfig.EmbeddingSize);
    public double[][] OutputW =  GenerateMatrix(TinyTransformerConfig.EmbeddingSize, TinyTransformerConfig.VocabSize);
    public double[][] ffn1 = GenerateMatrix(TinyTransformerConfig.EmbeddingSize, TinyTransformerConfig.EmbeddingSize*4);
    public double[][] ffn2 = GenerateMatrix(TinyTransformerConfig.EmbeddingSize*4, TinyTransformerConfig.EmbeddingSize);
    public double[] ffn1Bias = new double[TinyTransformerConfig.EmbeddingSize*4];
    public double[] ffn2Bias = new double[TinyTransformerConfig.EmbeddingSize];
    public double[] OutputBias = new double[TinyTransformerConfig.VocabSize];
    
    public static double[][] GenerateMatrix(int rows, int cols)
    {
        double[][] res = new double[rows][];
        
        for (int i = 0; i < rows; i++)
        {
            res[i] = new double[cols];
            
            for (int j = 0; j < cols; j++)
            {
                res[i][j] = rnd.NextDouble() * 0.2 - 0.1;
            }
        }

        return res;
    }
}