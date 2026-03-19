namespace Lib.Models.TinyTransformer.Configuration;

public class TinyTransformerConfig
{
        
    public double[][] TokenEmbeddings;
    public int VocabSize { get; set; }
    public int EmbeddingSize { get; set; }
    public int HeadCount { get; set; }
    public int ContextSize { get; set; }
    
    public TinyTransformerConfig( int VocabSize, int EmbeddingSize = 16, int HeadCount = 1, int ContextSize = 8 )
    {
        this.VocabSize =  VocabSize;
        this.EmbeddingSize = EmbeddingSize;
        this.HeadCount = HeadCount;
        this.ContextSize = ContextSize;
    }

}