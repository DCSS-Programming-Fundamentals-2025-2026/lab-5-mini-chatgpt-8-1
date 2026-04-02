using Lib.MathCore;
using Lib.Models.TinyTransformer.Configuration;
using Lib.Models.TinyTransformer.Layers;

namespace Lib.Models.TinyTransformer;

public class TinyTransformerModel
{
    private readonly TinyTransformerConfig _config;
    private readonly MathOpsImpl _mathOps = new ();
    public double[] NextTokenScores(int[] context)
    {
        TransformerBlock block = new TransformerBlock(_config);    
            
        return _mathOps.Softmax(block.Forward(context));
    }

    public TinyTransformerModel(TinyTransformerConfig config)
    {
        _config = config;
    }
}