using Lib.Models.TinyTransformer.Configuration;

namespace Lib.Models.TinyTransformer.State;

public class TinyTransformerWeights
{
    public static Random rnd = new Random(42);
    public double[][] wQ { get; set; }
    public double[][] wK { get; set; }
    public double[][] wV { get; set; }
    public double[][] wO { get; set; }
    public double[][] OutputW { get; set; }
    public double[][] ffn1 { get; set; }
    public double[][] ffn2 { get; set; }
    public double[] ffn1Bias { get; set; }
    public double[] ffn2Bias { get; set; }
    public double[] OutputBias { get; set; }

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

    public TinyTransformerWeights(int embeddingSize, int vocabSize)
    {
        wQ = GenerateMatrix(embeddingSize, embeddingSize);

        wK = GenerateMatrix(embeddingSize, embeddingSize);
        wV = GenerateMatrix(embeddingSize, embeddingSize);
        wO = GenerateMatrix(embeddingSize, embeddingSize);
        OutputW = GenerateMatrix(embeddingSize, vocabSize);

        ffn1 =
            GenerateMatrix(embeddingSize, embeddingSize * 4);

        ffn2 =
            GenerateMatrix(embeddingSize * 4, embeddingSize);

        ffn1Bias = new double[embeddingSize * 4];
        ffn2Bias = new double[embeddingSize];
        OutputBias = new double[vocabSize];
    }
}