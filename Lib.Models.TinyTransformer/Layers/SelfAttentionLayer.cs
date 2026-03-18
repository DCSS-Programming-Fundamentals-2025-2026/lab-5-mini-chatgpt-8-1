using Lib.Models.TinyTransformer.Configuration;

namespace Lib.Models.TinyTransformer.Layers;

public class SelfAttentionLayer
{
    public static double[][] Compute(int[] context, int embeddingSize)
    {
        double[][] x = new double[context.Length > 8 ? 8 : context.Length][];
        if (context.Length > 8)
        {
            context = context.TakeLast(8).ToArray();   
        }
        for (int i = 0; i < x.Length; i++)
        {
            x[i] = TinyTransformerConfig.TokenEmbeddings[context[i]];
        }
        
        return null;
    }
}