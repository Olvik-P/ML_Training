"""Агент-пылесос для уборки комнаты.

Демонстрирует цикл восприятие -> анализ -> действие (sense-act cycle).
Агент перемещается по двумерной комнате и собирает мусор.
"""

import random
import time

from constants import (
    ACTION_CLEAN,
    ACTION_MOVE,
    CLEAN,
    DEFAULT_DELAY,
    DEFAULT_STEPS,
    MOVEMENT_OFFSETS,
    RANDOM_SEED,
    ROOM_SIZE,
    SEPARATOR_WIDTH,
    STEPS,
    TRASH_VALUES,
)


class VacuumAgent:
    """Агент-пылесос, убирающий мусор в комнате."""

    def __init__(
        self,
        room: list[list[int]],
        start_row: int = 0,
        start_col: int = 0,
    ) -> None:
        """Инициализация агента.

        Args:
            room: Двумерный список, где 1 = мусор, 0 = чисто.
            start_row: Начальная строка (по умолчанию 0).
            start_col: Начальный столбец (по умолчанию 0).

        """
        self.room = room
        self.rows = len(room)
        self.cols = len(room[0]) if room else 0
        self.row = start_row
        self.col = start_col
        self.cleaned = CLEAN
        self.visited: set[tuple[int, int]] = set()
        self.visited.add((start_row, start_col))
        self.all_cells = {
            (row, col)
            for row in range(self.rows)
            for col in range(self.cols)
        }

    def perceive(self) -> int:
        """Восприятие: проверить, есть ли мусор в текущей клетке.

        Returns:
            TRASH если мусор есть, CLEAN если чисто.

        """
        return self.room[self.row][self.col]

    def _get_neighbours(self) -> list[tuple[int, int]]:
        """Вернуть список соседних клеток (вверх, вниз, влево, вправо).

        Returns:
            Список координат соседних клеток в пределах комнаты.

        """
        neighbours: list[tuple[int, int]] = []
        for dr, dc in MOVEMENT_OFFSETS:
            new_row = self.row + dr
            new_col = self.col + dc
            if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                neighbours.append((new_row, new_col))
        return neighbours

    def _choose_target(
        self,
        neighbours: list[tuple[int, int]],
        unvisited: set[tuple[int, int]],
    ) -> tuple[int, int]:
        """Выбрать целевую клетку для перемещения.

        Приоритеты:
            1. Непосещённый сосед.
            2. Ближайшая непосещённая клетка (манхэттенское расстояние).
            3. Любой сосед (все клетки посещены).

        Args:
            neighbours: Список соседних клеток.
            unvisited: Множество непосещённых клеток.

        Returns:
            Координаты целевой клетки.

        """
        unvisited_neighbours = [
            (r, c) for r, c in neighbours if (r, c) in unvisited
        ]

        if unvisited_neighbours:
            return random.choice(unvisited_neighbours)

        if unvisited:
            return min(
                unvisited,
                key=lambda cell: (
                    abs(cell[0] - self.row) + abs(cell[1] - self.col)
                ),
            )

        return random.choice(neighbours)

    def decide(
        self, has_trash: int,
    ) -> tuple[str, tuple[int, int] | None]:
        """Анализ: принять решение на основе восприятия.

        Args:
            has_trash: TRASH если есть мусор, CLEAN если чисто.

        Returns:
            Пара (действие, целевая клетка).

        """
        if has_trash:
            return ACTION_CLEAN, None

        neighbours = self._get_neighbours()
        unvisited = self.all_cells - self.visited
        target = self._choose_target(neighbours, unvisited)
        return ACTION_MOVE, target

    def act(
        self, decision: str, target: tuple[int, int] | None,
    ) -> str:
        """Действие: убрать мусор или переместиться.

        Args:
            decision: ACTION_CLEAN или ACTION_MOVE.
            target: Целевая клетка для перемещения (None для уборки).

        Returns:
            Сообщение о выполненном действии.

        """
        self.visited.add((self.row, self.col))
        if decision == ACTION_CLEAN:
            self.room[self.row][self.col] = CLEAN
            self.cleaned += 1
            return f'Убрал мусор в клетке ({self.row},{self.col})'

        # MOVE
        assert target is not None
        self.row, self.col = target
        return f'Переместился в ({self.row},{self.col})'

    def step(self) -> str:
        """Один цикл агента: восприятие -> анализ -> действие.

        Returns:
            Сообщение о выполненном действии.

        """
        trash_here = self.perceive()
        action, target = self.decide(trash_here)
        return self.act(action, target)

    def run(
        self, steps: int = DEFAULT_STEPS, delay: float = DEFAULT_DELAY,
    ) -> int:
        """Запустить агента на несколько шагов.

        Args:
            steps: Количество шагов.
            delay: Задержка между шагами в секундах.

        Returns:
            Количество убранного мусора.

        """
        print('=' * SEPARATOR_WIDTH)
        for i in range(steps):
            print(f'Шаг {i + 1}: {self.step()}')
            time.sleep(delay)
            if not (self.all_cells - self.visited):
                print(
                    f'\nОбследовал все {len(self.visited)} клеток комнаты!',
                )
                break
        print(
            f'\nУборка завершена. Убрано {self.cleaned} единиц мусора. '
            f'Не убрано {len(self.all_cells - self.visited)} клеток.',
        )
        return self.cleaned


def setup_room() -> list[list[int]]:
    """Создать и вывести карту комнаты с мусором.

    Returns:
        Двумерный список, где 1 = мусор, 0 = чисто.

    """
    random.seed(RANDOM_SEED)
    room_with_trash = [
        [random.choice(TRASH_VALUES) for _ in range(ROOM_SIZE)]
        for _ in range(ROOM_SIZE)
    ]

    print('Карта комнаты (1 = мусор, 0 = чисто):')
    for row in room_with_trash:
        print(row)

    return room_with_trash


# ------------------- ЗАПУСКАЕМ АГЕНТА -------------------
if __name__ == '__main__':
    agent = VacuumAgent(setup_room())
    print('Пылесос создан! Начинаю уборку...')
    cleaned_total = agent.run(steps=STEPS)
