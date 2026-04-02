using Lib.Models.TinyTransformer.Configuration;
using Lib.Models.TinyTransformer.Enums;
using Lib.Models.TinyTransformer.Factories;
using Lib.Models.TinyTransformer.Layers;
using NUnit.Framework;

namespace Lib.Models.TinyTransformer.Tests;

[TestFixture]
public class TinyTransformerTests
{
    public TinyTransformerConfig config;
    public SelfAttentionLayer selfAttentionLayer;
    public FeedForwardLayer feedForwardLayer;
    [SetUp]
    public void Setup()
    {
        float[][] tokenEmbeddings = new float[][]
        {
            new float[]
            {
                -0.0251f, 0.0901f, 0.0464f, 0.0197f, -0.0688f, -0.0688f, -0.0884f, 0.0732f,
                0.0112f, -0.0441f, 0.0821f, -0.0034f, 0.0556f, -0.0912f, 0.0223f, -0.0771f
            },
            new float[]
            {
                0.0202f, 0.0416f, -0.0959f, 0.0940f, 0.0665f, -0.0575f, -0.0636f, -0.0633f,
                -0.0128f, 0.0883f, -0.0334f, 0.0412f, -0.0091f, 0.0772f, -0.0551f, 0.0119f
            },
            new float[]
            {
                -0.0392f, 0.0050f, -0.0136f, -0.0418f, 0.0224f, -0.0721f, -0.0416f, -0.0267f,
                0.0911f, -0.0223f, 0.0554f, -0.0882f, 0.0129f, -0.0443f, 0.0667f, -0.0031f
            }
        };

        config = new(tokenEmbeddings.Length)
        {
            TokenEmbeddings = tokenEmbeddings
        };
        
        selfAttentionLayer = new SelfAttentionLayer(config);
        feedForwardLayer = new FeedForwardLayer(config);
    }

    [TestCase(new[] { 0, 2, 1 })]
    [TestCase(new[] { 1, 1, 1 })]
    [TestCase(new[] { 1, 0, 2 })]
    public void Test_Softmax_SumIsOne(int[] context)
    {
        float[][] x = selfAttentionLayer.InitXmatrix(context);

        float[][] Q = selfAttentionLayer.InitMatrix(x, QKV.Q);
        float[][] K = selfAttentionLayer.InitMatrix(x, QKV.K);
        float[][] V = selfAttentionLayer.InitMatrix(x, QKV.V);

        float[][] scores = MatrixHelper.MultiplyMatrix(Q, MatrixHelper.TransposeMatrix(K));
        selfAttentionLayer.EachElementDivideBySquareRootOfEmbeddingSizeWithMask(scores);

        float[][] attn = selfAttentionLayer.SoftmaxEachRow(scores);

        for (int i = 0; i < attn.Length; i++)
        {
            float sum = 0f;
            for (int j = 0; j < attn[i].Length; j++)
            {
                sum += attn[i][j];
            }

            if (Math.Abs(sum - 1.0) < 1e-12)
            {
                sum = 1;
            }

            Assert.That(sum, Is.EqualTo(1.0d));
        }
    }

    [TestCase(new[] { -0.123123f, 2.34827523f, -4.239785623f })]
    [TestCase(new[] { -0.000071562f, 0.00002836713f, -0.0000000623f })]
    [TestCase(new[] { -123123.123123f, 2.34827523f, -0.239785623f })]
    public void Test_Relu(float[] array)
    {
        FeedForwardLayer.Relu(array);
        
        for (int i = 0; i < array.Length; i++)
        {
            if (i % 2 == 0)
            {
                Assert.That(array[i], Is.EqualTo(0));
            }
            else
            {
                Assert.That(array[i], !Is.Negative);
            }
        }
    }
}