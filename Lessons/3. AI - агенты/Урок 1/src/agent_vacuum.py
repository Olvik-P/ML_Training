"""🤖 Агент-пылесос для уборки комнаты.

Демонстрирует цикл восприятие → анализ → действие (sense-act cycle).
Агент перемещается по двумерной комнате и собирает мусор.
"""

import random
import sys
import time

from constants import (
    ACTION_CLEAN,
    ACTION_MOVE,
    CLEAN,
    DEFAULT_DELAY,
    DEFAULT_STEPS,
    MOVEMENT_OFFSETS,
    SEPARATOR_WIDTH,
    STEP_FORWARD,
    VAL_MIN,
    RANDOM_SEED,
    ROOM_SIZE,
    STEPS,
    TRASH_VALUES
)

print(f"✅ Python версия: {sys.version_info.major}.{sys.version_info.minor}")
print("✅ Библиотеки готовы: random, time, sys — всё своё, без интернета")


class VacuumAgent:
    """Агент-пылесос, убирающий мусор в комнате."""

    def __init__(
        self,
        room: list[list[int]],
        start_row: int = CLEAN,
        start_col: int = CLEAN,
    ) -> None:
        """Инициализация агента.

        Args:
            room: Двумерный список, где 1 = мусор, 0 = чисто.
            start_row: Начальная строка (по умолчанию 0).
            start_col: Начальный столбец (по умолчанию 0).

        """
        self.room = room
        self.rows = len(room)
        self.cols = len(room[CLEAN]) if room else CLEAN
        self.row = start_row
        self.col = start_col
        self.cleaned = CLEAN
        print("🤖 Пылесос создан! Начинаю уборку...")

    def perceive(self) -> int:
        """Восприятие: проверить, есть ли мусор в текущей клетке.

        Returns:
            1 если мусор есть, 0 если чисто.

        """
        return self.room[self.row][self.col]

    @staticmethod
    def decide(has_trash: int) -> str:
        """Анализ: принять решение на основе восприятия.

        Args:
            has_trash: 1 если есть мусор, 0 если чисто.

        Returns:
            "CLEAN" если мусор есть, иначе "MOVE".

        """
        return ACTION_CLEAN if has_trash else ACTION_MOVE

    def _new_coordinate(self, val: int, max_val: int) -> int:
        """Вычислить новую координату со случайным смещением в соседнюю клетку.

        Агент не остаётся на месте — всегда двигается на ±1,
        если только не упёрся в границу (тогда только внутрь).

        Args:
            val: Текущая координата.
            max_val: Максимальное значение координаты (не включая).

        Returns:
            Новая координата в пределах [0, max_val).

        """
        if max_val <= STEP_FORWARD:
            return CLEAN  # некуда двигаться

        if val == VAL_MIN:
            return val + STEP_FORWARD  # только вправо/вниз
        if val == max_val - STEP_FORWARD:
            return val - STEP_FORWARD  # только влево/вверх
        return val + random.choice(MOVEMENT_OFFSETS)

    def act(self, decision: str) -> str:
        """Действие: убрать мусор или переместиться.

        Args:
            decision: "CLEAN" или "MOVE".

        Returns:
            Сообщение о выполненном действии.

        """
        if decision == ACTION_CLEAN:
            self.room[self.row][self.col] = CLEAN
            self.cleaned += STEP_FORWARD
            return f"🧹 Убрал мусор в клетке ({self.row},{self.col})"
        # MOVE
        self.row = self._new_coordinate(self.row, self.rows)
        self.col = self._new_coordinate(self.col, self.cols)
        return f"🚶‍♂️ Переместился в ({self.row},{self.col})"

    def step(self) -> str:
        """Один цикл агента: восприятие → анализ → действие.

        Returns:
            Сообщение о выполненном действии.

        """
        trash_here = self.perceive()
        action = self.decide(trash_here)
        return self.act(action)

    def run(
        self, steps: int = DEFAULT_STEPS, delay: float = DEFAULT_DELAY
    ) -> int:
        """Запустить агента на несколько шагов.

        Args:
            steps: Количество шагов.
            delay: Задержка между шагами в секундах.

        Returns:
            Количество убранного мусора.

        """
        print("=" * SEPARATOR_WIDTH)
        for i in range(steps):
            print(f"Шаг {i+1}: {self.step()}")
            time.sleep(delay)
        print(f"\n✅ Уборка завершена. Убрано {self.cleaned} единиц мусора.")
        return self.cleaned


def setup_room() -> list[list[int]]:
    """Создать и вывести карту комнаты с мусором.

    Returns:
        Двумерный список, где 1 = мусор, 0 = чисто.

    """
    # чтобы результат был одинаковым у всех (для проверки)
    random.seed(RANDOM_SEED)
    room_with_trash = [[random.choice(TRASH_VALUES) for _ in range(
        ROOM_SIZE)] for _ in range(ROOM_SIZE)]

    print("🗺️ Карта комнаты (1 = мусор, 0 = чисто):")
    for row in room_with_trash:
        print(row)

    return room_with_trash


# ------------------- ЗАПУСКАЕМ АГЕНТА -------------------
if __name__ == "__main__":
    agent = VacuumAgent(setup_room())
    cleaned_total = agent.run(steps=STEPS)
