namespace Lib.MathCore;

// Статичний клас, який служить головною точкою входу для всієї математики бібліотеки.
public static class MathOps
{
    // Створюємо один спільний екземпляр реалізації (MathOpsImpl).
    // Це економить пам'ять, бо нам не потрібно створювати новий об'єкт при кожному обчисленні.
    private static readonly IMathOps _default = new MathOpsImpl();

    // Властивість для доступу до математичних методів.
    // Інші розробники зможуть писати просто: MathOps.Default.Softmax(...)
    public static IMathOps Default => _default;
}