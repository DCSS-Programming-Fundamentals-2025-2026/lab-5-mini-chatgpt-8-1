namespace Lib.MathCore.Sampling;

public class ProbabilitySampler
{
    /// Вибирає індекс слова на основі ймовірностей (алгоритм "Колесо Фортуни").
    public int Sample(ReadOnlySpan<double> probs, Random rng)
    {
        // 1. Генеруємо випадкову "мішень" від 0.0 до 1.0.
        double target = rng.NextDouble();

        // 2. Змінна для накопичення ймовірностей.
        double cumulativeSum = 0.0;

        // 3. Йдемо по масиву, починаючи з САМОГО ПОЧАТКУ (індекс 0).
        for (int i = 0; i < probs.Length; i++)
        {
            // ВАЖЛИВО: додаємо поточну ймовірність до загальної суми.
            cumulativeSum += probs[i];

            // Якщо накопичена сума "перестрибнула" нашу мішень — ми знайшли слово!
            if (cumulativeSum > target)
            {
                return i;
            }
        }

        // 4. Якщо через мікроскопічні похибки (наприклад, сума вийшла 0.999999) 
        // ми пройшли весь цикл, просто повертаємо останній індекс.
        return probs.Length - 1;
    }
}