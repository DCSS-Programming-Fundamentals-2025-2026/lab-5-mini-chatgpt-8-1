using Lib.Models.TinyTransformer.Configuration;
using Lib.Models.TinyTransformer.Enums;
using Lib.Models.TinyTransformer.Layers;
using NUnit.Framework;

namespace Lib.Models.TinyTransformer.Tests;

[TestFixture]
public class TinyTransformerTests
{
    [SetUp]
    public void Setup()
    {
        TinyTransformerConfig.TokenEmbeddings = new double[][]
        {
            new double[]
            {
                -0.0251, 0.0901, 0.0464, 0.0197, -0.0688, -0.0688, -0.0884, 0.0732,
                0.0112, -0.0441, 0.0821, -0.0034, 0.0556, -0.0912, 0.0223, -0.0771
            },
            new double[]
            {
                0.0202, 0.0416, -0.0959, 0.0940, 0.0665, -0.0575, -0.0636, -0.0633,
                -0.0128, 0.0883, -0.0334, 0.0412, -0.0091, 0.0772, -0.0551, 0.0119
            },
            new double[]
            {
                -0.0392, 0.0050, -0.0136, -0.0418, 0.0224, -0.0721, -0.0416, -0.0267,
                0.0911, -0.0223, 0.0554, -0.0882, 0.0129, -0.0443, 0.0667, -0.0031
            }
        };

        TinyTransformerConfig config = new(TinyTransformerConfig.TokenEmbeddings.Length);
    }

    [TestCase(new[] { 0, 2, 1 })]
    [TestCase(new[] { 1, 1, 1 })]
    [TestCase(new[] { 1, 0, 2 })]
    public void Test_Softmax_SumIsOne(int[] context)
    {
        double[][] x = SelfAttentionLayer.InitXmatrix(context);

        double[][] Q = SelfAttentionLayer.InitMatrix(x, QKV.Q);
        double[][] K = SelfAttentionLayer.InitMatrix(x, QKV.K);
        double[][] V = SelfAttentionLayer.InitMatrix(x, QKV.V);

        double[][] scores = MatrixHelper.MultiplyMatrix(Q, MatrixHelper.TransposeMatrix(K));
        SelfAttentionLayer.EachElementDivideBySquareRootOfEmbeddingSizeWithMask(scores);

        double[][] attn = SelfAttentionLayer.SoftmaxEachRow(scores);

        for (int i = 0; i < attn.Length; i++)
        {
            double sum = 0d;
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

    [TestCase(new[] { -0.123123, 2.34827523, -4.239785623 })]
    [TestCase(new[] { -0.000071562, 0.00002836713, -0.0000000623 })]
    [TestCase(new[] { -123123.123123, 2.34827523, -0.239785623 })]
    public void Test_Relu(double[] array)
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