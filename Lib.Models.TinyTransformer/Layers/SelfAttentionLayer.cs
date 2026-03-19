using Lib.MathCore;
using Lib.Models.TinyTransformer.Configuration;
using Lib.Models.TinyTransformer.Enums;
using Lib.Models.TinyTransformer.State;


namespace Lib.Models.TinyTransformer.Layers;

public class SelfAttentionLayer
{
    public static TinyTransformerWeights weights;
    public static TinyTransformerConfig config;
    
    public static MathOpsImpl mathOps = new MathOpsImpl();
    
    public static double[] Compute(int[] context)
    {
        //step 1
        double[][] x = InitXmatrix(context);

        //step 2
        double[][] Q = InitMatrix(x, QKV.Q);
        double[][] K = InitMatrix(x, QKV.K);
        double[][] V = InitMatrix(x, QKV.V);
        
        //step 3
        double[][] scores = MultiplyMatrix(Q, TransposeMatrix(K));
        EachElementDivideBySquareRootOfEmbeddingSizeWithMask(scores);
        
        //step 4
        double[][] attn = SoftmaxEachRow(scores);
        
        //step 5
        double[][] outMatrix = WeightedSum(attn, V);
        
        //step 6
        double[][] proj = MultiplyMatrix(outMatrix, weights.wO);

        //step 7
        return proj[proj.Length - 1];
    }


    public static double[][] InitXmatrix(int[] context)
    {
        double[][] x = new double[context.Length > config.ContextSize ? config.ContextSize : context.Length][];

        if (context.Length > 8)
        {
            context = context.TakeLast(8).ToArray();
        }

        for (int i = 0; i < x.Length; i++)
        {
            x[i] = config.TokenEmbeddings[context[i]];
        }

        return x;
    }
    
    public static double[][] InitMatrix(double[][] x, QKV qkv)
    {
        switch (qkv)
        {
            case QKV.K:
                return MultiplyMatrix(x, weights.wK);
            case QKV.Q:
                return MultiplyMatrix(x, weights.wQ);
            case QKV.V:
                return MultiplyMatrix(x, weights.wV);
            default:
                throw new ArgumentOutOfRangeException(nameof(qkv), qkv, null);
        }
    }

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
    
    public static void EachElementDivideBySquareRootOfEmbeddingSizeWithMask(double[][] matrix)
    {
        for (int i = 0; i < matrix.Length; i++)
        {
            for (int j = 0; j < matrix[i].Length; j++)
            {
                if (j <= i)
                {
                    matrix[i][j] /= Math.Sqrt(config.EmbeddingSize);
                    continue;
                }

                matrix[i][j] = double.NegativeInfinity;
            }
        }
    }
    
    public static double[][] SoftmaxEachRow(double[][] matrix)
    {
        double[][] res = new double[matrix.Length][];
        
        for (int i = 0; i < matrix.Length; i++)
        {
            res[i] = mathOps.Softmax(matrix[i]);
        }

        return res;
    }
    
    public static double[][] TransposeMatrix(double[][] matrix)
    {
        double[][] res = new double[matrix.GetLength(1)][];

        for (int i = 0; i < res.Length; i++)
        {
            res[i] = new double[matrix.GetLength(0)];

            for (int j = 0; j < res[i].Length; j++)
            {
                res[i][j] = matrix[j][i];
            }
        }

        return res;
    }
    
    public static double[][] WeightedSum(double[][] attn, double[][] V)
    {
        double[][] res = new double[attn.Length][];
        
        for (int i = 0; i < attn.Length; i++) 
        {
            double[] sumVector = new double[config.EmbeddingSize];
            
            for (int j = 0; j < attn[i].Length; j++) 
            {
                double weight = attn[i][j];
                double[] valueVect = V[j];
        
                for (int k = 0; k < config.EmbeddingSize; k++) 
                {
                    sumVector[k] += weight * valueVect[k];
                }
            }
            res[i] = sumVector;
        }
        
        return res;
    }

}
