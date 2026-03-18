using Lib.Models.TinyTransformer.Configuration;
using Lib.Models.TinyTransformer.Layers;

namespace Lib.Models.TinyTransformer;

public class TinyTransformerModel
{
    public double[] NextTokenScores(int[] context)
    {
        return TransformerBlock.Forward(context, TinyTransformerConfig.VocabSize, TinyTransformerConfig.EmbeddingSize);
    }
}