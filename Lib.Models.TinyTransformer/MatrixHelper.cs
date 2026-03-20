namespace Lib.Models.TinyTransformer;

public class MatrixHelper
{
    public static double[][] MultiplyMatrix(double[][] matrix, double[][] weights)
    {

        if (matrix[0].Length != weights.Length)
        {
            throw new ArgumentException("Matrix dimensions do not match");
        }
        
        double[][] res = new double[matrix.Length][];

        for (int i = 0; i < res.Length; i++)
        {
            res[i] = new double[weights[0].Length];

            for (int j = 0; j < res[i].Length; j++)
            {
                double sum = 0;

                for (int k = 0; k < matrix.Length; k++)
                {
                    sum += matrix[i][k] * weights[k][j];
                }
                
                res[i][j] = sum;
            }
        }

        return res;
    }
    
    public static double[][] TransposeMatrix(double[][] matrix)
    {
        double[][] res = new double[matrix[0].Length][];

        for (int i = 0; i < res.Length; i++)
        {
            res[i] = new double[matrix.Length];

            for (int j = 0; j < res[i].Length; j++)
            {
                res[i][j] = matrix[j][i];
            }
        }

        return res;
    }

    public static void LineSumm(double[] line1 , double[] line2)
    {
        if (line1.Length != line2.Length)
        {
            throw new IndexOutOfRangeException();
        }

        for (int i = 0; i < line1.Length; i++)
        {
            line1[i] += line2[i];
        }
    }
}