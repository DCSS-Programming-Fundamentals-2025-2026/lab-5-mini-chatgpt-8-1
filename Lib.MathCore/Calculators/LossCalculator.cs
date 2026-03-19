namespace Lib.MathCore.Calculators;

public class LossCalculator
{
    private readonly SoftmaxCalculator _softmax = new();

    public double Calculate(ReadOnlySpan<double> logits, int target)
    {
        // Крок 1: Отримуємо масив усіх ймовірностей від нашого Softmax
        double[] probabilities = _softmax.Calculate(logits);

        // Крок 2: Вибираємо з цього масиву ТІЛЬКИ ОДНЕ число — 
        // ймовірність того слова, яке ми реально очікували (target).
        double p = probabilities[target];

        // Крок 3: Рахуємо логарифмічну помилку.
        // Чим ближче p до 1.0, тим меншим буде результат (помилка).
        // Додаємо 1e-10, щоб програма не "вибухнула", якщо p раптом дорівнює 0.
        return -Math.Log(p + 1e-10);
    }
}