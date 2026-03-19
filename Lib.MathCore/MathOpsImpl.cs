// Підключаємо наші папки (namespaces), щоб MathOpsImpl бачив калькулятори
using Lib.MathCore.Calculators;
using Lib.MathCore.Sampling;
using Lib.MathCore.Utilities;

namespace Lib.MathCore;

public class MathOpsImpl : IMathOps
{
    // Створюємо екземпляри наших спеціалізованих класів
    private readonly SoftmaxCalculator _softmax = new();
    private readonly LossCalculator _loss = new();
    private readonly ProbabilitySampler _sampler = new();

    // 1. Реалізація Softmax через калькулятор
    public double[] Softmax(ReadOnlySpan<double> logits)
    {
        return _softmax.Calculate(logits);
    }

    // 2. Реалізація CrossEntropyLoss через калькулятор
    public double CrossEntropyLoss(ReadOnlySpan<double> logits, int target)
    {
        return _loss.Calculate(logits, target);
    }

    // 3. Реалізація ArgMax через Utility-клас
    public int ArgMax(ReadOnlySpan<double> scores)
    {
        return ScoreUtilities.GetArgMax(scores);
    }

    // 4. Реалізація SampleFromProbs через Sampler
    public int SampleFromProbs(ReadOnlySpan<double> probs, Random rng)
    {
        return _sampler.Sample(probs, rng);
    }
}