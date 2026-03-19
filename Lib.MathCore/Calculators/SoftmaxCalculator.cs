namespace Lib.MathCore.Calculators;

public class SoftmaxCalculator
{
    public double[] Calculate(ReadOnlySpan<double> logits)
    {
        // Перевірка на порожній вхід
        if (logits.Length == 0)
        {
            return Array.Empty<double>();
        }

        // --- КРОК 1: ШУКАЄМО МАКСИМУМ ---
        // Це потрібно для "числової стабільності". 
        // Без цього Exp(1000) видасть помилку нескінченності (Infinity).
        double max = logits[0];
        for (int i = 1; i < logits.Length; i++)
        {
            if (logits[i] > max)
            {
                max = logits[i];
            }
        }

        // --- КРОК 2: ОБЧИСЛЮЄМО ЕКСПОНЕНТИ ТА ЇХНЮ СУМУ ---
        // Створюємо масив для результатів
        double[] exp = new double[logits.Length];
        double sum = 0;

        for (int i = 0; i < logits.Length; i++)
        {
            // Віднімаємо максимум (max) від кожного значення. 
            // Тепер найбільше число стане 0 (Exp(0) = 1), а інші — від'ємними.
            exp[i] = Math.Exp(logits[i] - max);
            
            // Одразу накопичуємо суму всіх отриманих значень
            sum += exp[i];
        }

        // --- КРОК 3: НОРМАЛІЗАЦІЯ ---
        // Ділимо кожну експоненту на загальну суму.
        for (int i = 0; i < exp.Length; i++)
        {
            // Використовуємо наш улюблений оператор /= для перезапису
            exp[i] /= sum;
        }

        // Тепер у нас в масиві exp сума всіх чисел дорівнює рівно 1.0!
        return exp;
    }
}