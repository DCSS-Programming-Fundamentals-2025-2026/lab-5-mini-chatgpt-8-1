using Lib.Models.TinyTransformer.Configuration;
using Microsoft.VisualBasic.CompilerServices;

namespace Lib.Models.TinyTransformer.Layers;

public class TransformerBlock
{
    public static double[] Forward(int[] context)
    {
        return FeedForwardLayer.Project(SelfAttentionLayer.Compute(context));
    }
}