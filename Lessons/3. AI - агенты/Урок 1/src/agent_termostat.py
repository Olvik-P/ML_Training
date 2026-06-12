import sys
import time

print(f"✅ Python версия: {sys.version_info.major}.{sys.version_info.minor}")
print("✅ Библиотеки готовы: random, time, sys — всё своё, без интернета")


test_temperatures = [18, 20, 22, 23, 21, 15, 25, 22]


class ThermostatAgent:
    """Агент для управления температурой в комнате."""

    def __init__(self, target_temp: int = 22.0) -> None:
        """Инициализация агента."""
        self.target_temp = target_temp  # комфортная температура
        self.heater_on = False  # состояние обогревателя
        print(f"🌡️ Агент создан. Цель: {self.target_temp}°C")

    def perceive(self, current_temp: int) -> int:
        """Шаг 1: Восприятие — агент узнаёт температуру."""
        self.current_temp = current_temp
        return self.current_temp

    def decide(self) -> str:
        """Шаг 2: Анализ — принимает решение по правилам."""
        if self.current_temp < self.target_temp - 1:
            return "HEAT_ON"  # слишком холодно → включить
        if self.current_temp > self.target_temp + 1:
            return "HEAT_OFF"  # слишком жарко → выключить
        return "DO_NOTHING"  # нормально

    def act(self, decision: str) -> str:
        """Шаг 3: Действие — выполняет решение."""
        if decision == "HEAT_ON" and not self.heater_on:
            self.heater_on = True
            return "🔥 Включаю обогрев"
        if decision == "HEAT_OFF" and self.heater_on:
            self.heater_on = False
            return "❄️ Выключаю обогрев"
        if decision == "DO_NOTHING":
            return "😌 Всё хорошо, ничего не делаю"
        return "⚙️ Без изменений"

    def step(self, current_temp: int) -> str:
        """Один полный цикл агента: восприятие → анализ → действие."""
        self.perceive(current_temp)
        decision = self.decide()
        action_message = self.act(decision)
        return f"Температура: {current_temp}°C → {action_message}"


# ------------------- ЗАПУСКАЕМ АГЕНТА -------------------
def main() -> None:
    """Запуск агента."""
    print("\n" + "=" * 50)
    agent = ThermostatAgent(target_temp=22)

    # Симулируем изменение температуры в комнате

    print("\n🏠 Комната начинает остывать и нагреваться...\n")
    for temp in test_temperatures:
        result = agent.step(temp)
        print(result)
        time.sleep(0.5)  # маленькая пауза, чтобы было красиво

    print(
        "\n✅ Агент отработал все циклы. Видите? "
        "Простые правила → умное поведение."
    )


if __name__ == "__main__":
    main()
