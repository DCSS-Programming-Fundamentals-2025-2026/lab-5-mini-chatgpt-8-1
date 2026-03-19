namespace Lib.Models.TinyTransformer.State;

public class TinyTransformerWeights
{
    public static Random rnd = new Random(42);
    
    public double[][] wQ = GenerateMatrix(16, 16);
    public double[][] wK = GenerateMatrix(16, 16);
    public double[][] wV = GenerateMatrix(16, 16);
    public double[][] wO = GenerateMatrix(16, 16);
    
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