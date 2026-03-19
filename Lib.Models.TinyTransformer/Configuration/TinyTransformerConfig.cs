namespace Lib.Models.TinyTransformer.Configuration;

public class TinyTransformerConfig
{
    public static double[][] TokenEmbeddings;
    public static int VocabSize { get; set; }
    public static int EmbeddingSize { get; set; }
    public static int HeadCount { get; set; }
    public static int ContextSize { get; set; }
    
    public TinyTransformerConfig( int vocabSize, int embeddingSize = 16, int headCount = 1, int contextSize = 8 )
    {
        VocabSize =  vocabSize;
        EmbeddingSize = embeddingSize;
        HeadCount = headCount;
        ContextSize = contextSize;
    }

}