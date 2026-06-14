"""Модуль для синхронной работы нескольких агентов-пылесосов.

Особенности:
- Агенты не видят мусор заранее — только попадая в клетку
- Синхронное выполнение — агенты ходят по очереди
- Общая комната с блокировками для безопасности
- Потокобезопасная регистрация и перемещение агентов
"""

import random
import threading
import time
from dataclasses import dataclass, field

from constants import (
    ACTION_CLEAN,
    ACTION_MOVE,
    CLEAN,
    MOVEMENT_OFFSETS,
    SEPARATOR_WIDTH,
    TRASH,
)

# ──────────────────────────────────────────────
# Конфигурация симуляции
# ──────────────────────────────────────────────


@dataclass
class SimulationConfig:
    """Конфигурация симуляции уборки комнаты."""

    room_size: int = 5
    num_agents: int = 2
    max_steps: int = 30
    seed: int = 42
    step_delay: float = 0.1

    # Стартовые позиции для агентов (по углам)
    start_positions: tuple[tuple[int, int], ...] = field(
        default_factory=lambda: (
            (0, 0),
            (0, 4),
            (4, 0),
            (4, 4),
        )
    )

    def __post_init__(self) -> None:
        """Валидация конфигурации после инициализации."""
        if self.room_size < 2:
            raise ValueError("Размер комнаты должен быть >= 2")
        if self.num_agents < 1:
            raise ValueError("Должен быть хотя бы 1 агент")
        if self.max_steps < 1:
            raise ValueError("Максимальное количество шагов должно быть >= 1")


# ──────────────────────────────────────────────
# Комната с общей памятью
# ──────────────────────────────────────────────


class SharedRoom:
    """Комната с общей памятью и потокобезопасными операциями.

    Хранит сетку с мусором и позиции всех агентов.
    Все операции чтения/записи защищены блокировкой (threading.Lock).
    """

    def __init__(self, size: int, seed: int = 42) -> None:
        """Инициализировать комнату с сеткой мусора.

        Args:
            size: Размер комнаты (size x size).
            seed: Сид генератора случайных чисел.

        """
        random.seed(seed)
        self.size = size
        self.grid = [
            [random.choice([CLEAN, TRASH]) for _ in range(size)]
            for _ in range(size)
        ]
        self.lock = threading.Lock()
        self.positions: dict[int, tuple[int, int]] = {}
        self.total_cleaned = 0

    def has_trash(self, row: int, col: int) -> bool:
        """Проверить наличие мусора в клетке (только чтение).

        Args:
            row: Номер строки.
            col: Номер столбца.

        Returns:
            True, если в клетке есть мусор.

        """
        with self.lock:
            return self.grid[row][col] == TRASH

    def try_clean(self, row: int, col: int, agent_id: int) -> bool:
        """Атомарно убрать мусор из клетки.

        Args:
            row: Номер строки.
            col: Номер столбца.
            agent_id: Идентификатор агента.

        Returns:
            True, если мусор был успешно убран.

        """
        with self.lock:
            if self.grid[row][col] == TRASH:
                self.grid[row][col] = CLEAN
                self.total_cleaned += 1
                return True
            return False

    def register_agent(self, agent_id: int, row: int, col: int) -> None:
        """Потокобезопасно зарегистрировать агента в комнате.

        Args:
            agent_id: Идентификатор агента.
            row: Начальная строка.
            col: Начальный столбец.

        """
        with self.lock:
            self.positions[agent_id] = (row, col)

    def try_move(self, agent_id: int, new_row: int, new_col: int) -> bool:
        """Атомарно переместить агента, если клетка свободна.

        Args:
            agent_id: Идентификатор агента.
            new_row: Новая строка.
            new_col: Новый столбец.

        Returns:
            True, если перемещение успешно.

        """
        if not (0 <= new_row < self.size and 0 <= new_col < self.size):
            return False

        with self.lock:
            if (new_row, new_col) in self.positions.values():
                return False
            self.positions[agent_id] = (new_row, new_col)
            return True

    def get_position(self, agent_id: int) -> tuple[int, int]:
        """Получить позицию агента.

        Args:
            agent_id: Идентификатор агента.

        Returns:
            Кортеж (row, col) с координатами агента.

        """
        with self.lock:
            return self.positions.get(agent_id, (0, 0))

    def print_map(self) -> None:
        """Вывести карту комнаты (для отладки)."""
        with self.lock:
            print("Текущая карта комнаты (1=мусор, 0=чисто):")
            for row in self.grid:
                print(row)


# ──────────────────────────────────────────────
# Агент-пылесос
# ──────────────────────────────────────────────


class SharedVacuumAgent:
    """Агент-пылесос, работающий в общей комнате.

    Агент исследует комнату, убирая мусор и запоминая посещённые клетки.
    Решение о следующем действии принимается на основе восприятия и памяти.
    """

    def __init__(
        self,
        agent_id: int,
        room: SharedRoom,
        start_row: int,
        start_col: int,
    ) -> None:
        """Инициализировать агента-пылесоса.

        Args:
            agent_id: Уникальный идентификатор агента.
            room: Общая комната, в которой работает агент.
            start_row: Начальная строка.
            start_col: Начальный столбец.

        """
        self.id = agent_id
        self.room = room
        self.rows = room.size
        self.cols = room.size

        # Потокобезопасная регистрация в общей комнате
        room.register_agent(agent_id, start_row, start_col)

        # Локальная память агента
        self.visited: set[tuple[int, int]] = set()
        self.visited.add((start_row, start_col))
        self.all_cells: set[tuple[int, int]] = {
            (r, c) for r in range(self.rows) for c in range(self.cols)
        }
        self.cleaned = 0

    def perceive(self) -> int:
        """Восприятие — проверка мусора в текущей клетке.

        Returns:
            TRASH (1), если в клетке мусор, иначе CLEAN (0).

        """
        row, col = self.room.get_position(self.id)
        return TRASH if self.room.has_trash(row, col) else CLEAN

    def _get_neighbours(self) -> list[tuple[int, int]]:
        """Получить список соседних (доступных) клеток.

        Returns:
            Список координат соседних клеток в пределах комнаты.

        """
        row, col = self.room.get_position(self.id)
        neighbours: list[tuple[int, int]] = []
        for dr, dc in MOVEMENT_OFFSETS:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                neighbours.append((nr, nc))
        return neighbours

    def _choose_target(
        self,
        neighbours: list[tuple[int, int]],
        unvisited: set[tuple[int, int]],
    ) -> tuple[int, int]:
        """Выбрать целевую клетку на основе памяти.

        Стратегия выбора:
        1. Случайная непосещённая соседняя клетка
        2. Ближайшая непосещённая клетка (по Манхэттенскому расстоянию)
        3. Случайная соседняя клетка

        Args:
            neighbours: Список соседних клеток.
            unvisited: Множество всех непосещённых клеток.

        Returns:
            Координаты выбранной целевой клетки.

        """
        unvisited_neighbours = [n for n in neighbours if n in unvisited]
        if unvisited_neighbours:
            return random.choice(unvisited_neighbours)

        if unvisited:
            row, col = self.room.get_position(self.id)
            return min(
                unvisited,
                key=lambda cell: abs(cell[0] - row) + abs(cell[1] - col),
            )

        return random.choice(neighbours)

    def decide(self, has_trash: int) -> tuple[str, tuple[int, int] | None]:
        """Принять решение на основе восприятия.

        Args:
            has_trash: Результат восприятия (TRASH или CLEAN).

        Returns:
            Кортеж (действие, цель).

        """
        if has_trash == TRASH:
            return ACTION_CLEAN, None

        neighbours = self._get_neighbours()
        unvisited = self.all_cells - self.visited
        return ACTION_MOVE, self._choose_target(neighbours, unvisited)

    def act(self, decision: str, target: tuple[int, int] | None) -> str:
        """Выполнить действие и вернуть отчёт.

        Args:
            decision: Решение агента (ACTION_CLEAN или ACTION_MOVE).
            target: Целевая клетка для перемещения (может быть None).

        Returns:
            Строка с отчётом о выполненном действии.

        """
        row, col = self.room.get_position(self.id)
        self.visited.add((row, col))

        if decision == ACTION_CLEAN:
            if self.room.try_clean(row, col, self.id):
                self.cleaned += 1
                return f"[{self.id}] 🧹 Убрал в ({row},{col})"
            return f"[{self.id}] ⚠️ ({row},{col}) уже чисто"

        if target is not None and self.room.try_move(
            self.id, target[0], target[1]
        ):
            return f"[{self.id}] 🚶 Перешёл в {target}"
        return (
            f"[{self.id}] 🚫 {target} занят, остаюсь в ({row},{col})"
        )

    def step(self) -> str:
        """Один цикл восприятие → решение → действие (sense-act).

        Returns:
            Строка с отчётом о выполненном шаге.

        """
        trash = self.perceive()
        action, target = self.decide(trash)
        return self.act(action, target)

    def is_done(self) -> bool:
        """Проверить, исследовал ли агент все клетки.

        Returns:
            True, если все клетки посещены.

        """
        return len(self.visited) == len(self.all_cells)


# ──────────────────────────────────────────────
# Симуляция
# ──────────────────────────────────────────────


def run_simulation(config: SimulationConfig | None = None) -> None:
    """Запуск синхронной симуляции уборки комнаты.

    Агенты действуют по очереди (синхронно) в общей комнате.
    Симуляция завершается, когда все агенты исследовали все клетки
    или когда достигнуто максимальное количество шагов.

    Args:
        config: Конфигурация симуляции. Если None, используются
                значения по умолчанию.

    """
    if config is None:
        config = SimulationConfig()

    # Создаём комнату
    room = SharedRoom(config.room_size, config.seed)
    room.print_map()

    # Создаём агентов
    agents: list[SharedVacuumAgent] = []
    for i in range(config.num_agents):
        row, col = config.start_positions[i % len(config.start_positions)]
        agent = SharedVacuumAgent(i, room, row, col)
        agents.append(agent)
        print(f"🤖 Агент {i} создан на позиции ({row},{col})")

    print("\n" + "=" * SEPARATOR_WIDTH)
    print("НАЧАЛО СИНХРОННОЙ СИМУЛЯЦИИ")
    print("Агенты ходят ПО ОЧЕРЕДИ (синхронно)")
    print("=" * SEPARATOR_WIDTH + "\n")

    # Основной цикл симуляции
    for step in range(config.max_steps):
        print(f"--- Шаг {step + 1} ---")

        all_done = True
        for agent in agents:
            result = agent.step()
            print(f"  {result}")
            time.sleep(config.step_delay)

            if not agent.is_done():
                all_done = False

        if all_done:
            print(
                f"\n✨ Все агенты исследовали всю комнату на шаге {step + 1}!"
            )
            break
        print()

    # Финальная статистика
    print("\n" + "=" * SEPARATOR_WIDTH)
    print("📊 ИТОГИ СИМУЛЯЦИИ")
    print("=" * SEPARATOR_WIDTH)
    print(f"Всего убрано мусора: {room.total_cleaned}")

    for agent in agents:
        visited_percent = len(agent.visited) / len(agent.all_cells) * 100
        print(
            f"Агент {agent.id}: убрал {agent.cleaned} ед., "
            f"посетил {len(agent.visited)}/{len(agent.all_cells)} "
            f"клеток ({visited_percent:.0f}%)"
        )

    room.print_map()


if __name__ == "__main__":
    run_simulation(SimulationConfig())
