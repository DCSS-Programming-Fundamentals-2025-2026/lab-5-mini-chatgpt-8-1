namespace Lib.Models.TinyTransformer.Configuration;

public class TinyTransformerConfig
{
    public static double[][] TokenEmbeddings;
    public static int VocabSize => TokenEmbeddings.Length;
    public static int EmbeddingSize = 16;
}