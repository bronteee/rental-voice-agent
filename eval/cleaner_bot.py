from __future__ import annotations


class DeterministicCleanerBot:
    def __init__(self, cleaner_turns: dict[int, str]) -> None:
        self.cleaner_turns = cleaner_turns

    def next_response(self, agent_turn_index: int) -> str | None:
        return self.cleaner_turns.get(agent_turn_index)
