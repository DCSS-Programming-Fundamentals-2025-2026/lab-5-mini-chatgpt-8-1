using Lib.MathCore;
using NUnit.Framework; // Бібліотека NUnit для тестування

namespace Lib.MathCore.Tests;

// Помічаємо весь клас як набір тестів (Fixture)
[TestFixture] 
public class MathCoreTests
{
    // ТЕСТ 1: Перевірка пошуку максимального значення (ArgMax)
    [Test]
    public void Test_ArgMax_FindsMaximum()
    {
        // 1. Готуємо вхідні дані: число 5.0 є найбільшим і стоїть під індексом 2
        double[] scores = { 0.5f, 1.2f, 5.0f, 0.8f }; 
        
        // 2. Викликаємо метод через нашу точку доступу Default
        int result = MathOps.Default.ArgMax(scores);
        
        // 3. Перевіряємо, чи повернув метод правильний індекс (має бути 2)
        Assert.That(result, Is.EqualTo(2)); 
    }

    // ТЕСТ 2: Перевірка нормалізації ймовірностей (Softmax)
    [Test]
    public void Test_Softmax_SumIsOne()
    {
        // 1. Готуємо довільні логіти
        double[] logits = { 1.0f, 2.0f, 3.0f };
        
        // 2. Отримуємо ймовірності після обробки Softmax
        double[] probs = MathOps.Default.Softmax(logits);
        
        // 3. Рахуємо загальну суму всіх отриманих чисел
        double sum = 0;
        foreach (var p in probs) sum += p;

        // 4. Перевіряємо, що сума дорівнює рівно 1.0 (100%).
        // Додаємо Within(0.0001f), бо у double бувають мікро-похибки при діленні.
        Assert.That(sum, Is.EqualTo(1.0f).Within(0.0001f));
    }

    // ТЕСТ 3: Перевірка розрахунку помилки (CrossEntropyLoss)
    [Test]
    public void Test_Loss_CorrectPrediction()
    {
        // 1. Створюємо ситуацію, де модель майже впевнена в правильній відповіді
        // (перше число 20.0 набагато більше за інші)
        double[] logits = { 20.0f, 0.0f, 0.0f }; 
        int target = 0; // Кажемо, що правильне слово — під індексом 0

        // 2. Обчислюємо помилку (чим краще вгадали, тим меншим має бути Loss)
        double loss = MathOps.Default.CrossEntropyLoss(logits, target);

        // 3. Перевіряємо, що помилка дуже маленька (майже нуль)
        Assert.That(loss, Is.LessThan(0.001f));
    }

    // ТЕСТ 4: Перевірка випадкового вибору (SampleFromProbs)
    [Test]
    public void Test_Sample_ReturnsValidIndex()
    {
        // 1. Створюємо масив ймовірностей
        double[] probs = { 0.1f, 0.7f, 0.2f };
        Random rng = new Random();

        // 2. Просимо метод вибрати випадковий індекс 100 разів поспіль
        for (int i = 0; i < 100; i++)
        {
            int result = MathOps.Default.SampleFromProbs(probs, rng);
            
            // 3. Перевіряємо, що кожен раз індекс лежить у межах нашого масиву (від 0 до 2)
            Assert.That(result, Is.GreaterThanOrEqualTo(0));
            Assert.That(result, Is.LessThan(probs.Length));
        }
    }
}