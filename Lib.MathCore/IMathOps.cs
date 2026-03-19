namespace Lib.MathCore;

public interface IMathOps
{
    // Перетворює сирі оцінки (логіти) у ймовірності. 
    // Сума всіх значень у результаті завжди дорівнює 1.0.
    double[] Softmax(ReadOnlySpan<double> logits);

    // Обчислює помилку моделі (Loss) для конкретного слова. 
    // Використовується при тренуванні мережі.
    double CrossEntropyLoss(ReadOnlySpan<double> logits, int target);

    // Знаходить індекс найбільшого числа в масиві. 
    // Потрібен для вибору найбільш ймовірного наступного слова.
    int ArgMax(ReadOnlySpan<double> scores);

    // Вибирає випадковий індекс на основі отриманих ймовірностей. 
    // Додає різноманітності у відповіді чату.
    int SampleFromProbs(ReadOnlySpan<double> probs, Random rng);
}