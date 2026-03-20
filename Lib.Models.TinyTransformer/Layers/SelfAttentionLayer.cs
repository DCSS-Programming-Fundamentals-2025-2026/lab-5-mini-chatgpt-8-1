using Lib.MathCore;
using Lib.Models.TinyTransformer.Configuration;
using Lib.Models.TinyTransformer.Enums;
using Lib.Models.TinyTransformer.State;


namespace Lib.Models.TinyTransformer.Layers;

public class SelfAttentionLayer
{
    public static TinyTransformerWeights Weights = new();
    public static MathOpsImpl MathOps = new ();
    
    public static double[] Compute(int[] context)
    {
        //step 1
        double[][] x = InitXmatrix(context);

        //step 2
        double[][] Q = InitMatrix(x, QKV.Q);
        double[][] K = InitMatrix(x, QKV.K);
        double[][] V = InitMatrix(x, QKV.V);
        
        //step 3
        double[][] scores = MatrixHelper.MultiplyMatrix(Q, MatrixHelper.TransposeMatrix(K));
        EachElementDivideBySquareRootOfEmbeddingSizeWithMask(scores);
        
        //step 4
        double[][] attn = SoftmaxEachRow(scores);
        
        //step 5
        double[][] outMatrix = WeightedSum(attn, V);
        
        //step 6
        double[][] proj = MatrixHelper.MultiplyMatrix(outMatrix, Weights.wO);

        //step 7
        return proj[proj.Length - 1];
    }

    public static double[][] InitXmatrix(int[] context)
    {
        double[][] x = new double[context.Length > TinyTransformerConfig.ContextSize ? TinyTransformerConfig.ContextSize : context.Length][];

        if (context.Length > 8)
        {
            context = context.TakeLast(8).ToArray();
        }

        for (int i = 0; i < x.Length; i++)
        {
            x[i] = TinyTransformerConfig.TokenEmbeddings[context[i]];
        }

        return x;
    }
    
    public static double[][] InitMatrix(double[][] x, QKV qkv)
    {
        switch (qkv)
        {
            case QKV.K:
                return MatrixHelper.MultiplyMatrix(x, Weights.wK);
            case QKV.Q:
                return MatrixHelper.MultiplyMatrix(x, Weights.wQ);
            case QKV.V:
                return MatrixHelper.MultiplyMatrix(x, Weights.wV);
            default:
                throw new ArgumentOutOfRangeException(nameof(qkv), qkv, null);
        }
    }
    
    public static void EachElementDivideBySquareRootOfEmbeddingSizeWithMask(double[][] matrix)
    {
        for (int i = 0; i < matrix.Length; i++)
        {
            for (int j = 0; j < matrix[i].Length; j++)
            {
                if (j <= i)
                {
                    matrix[i][j] /= Math.Sqrt(TinyTransformerConfig.EmbeddingSize);
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
            res[i] = MathOps.Softmax(matrix[i]);
        }

        return res;
    }
    
    public static double[][] WeightedSum(double[][] attn, double[][] V)
    {
        double[][] res = new double[attn.Length][];
        
        for (int i = 0; i < attn.Length; i++) 
        {
            double[] sumVector = new double[TinyTransformerConfig.EmbeddingSize];
            
            for (int j = 0; j < attn[i].Length; j++) 
            {
                double weight = attn[i][j];
                double[] valueVect = V[j];
        
                for (int k = 0; k < TinyTransformerConfig.EmbeddingSize; k++) 
                {
                    sumVector[k] += weight * valueVect[k];
                }
            }
            res[i] = sumVector;
        }
        
        return res;
    }
}
