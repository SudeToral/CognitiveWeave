"""AgentPopulation — manages a group of SocietyAgents and runs experiment cycles.

The population enforces the bandwidth constraint: agents run sequentially
within a cycle so each agent reads the graph state as it was after the
previous agent's write. This creates a simple causal chain — Agent B can
read what Agent A just wrote if it's relevant to B's interest.

Each full cycle produces a list of CycleWrite records, one per agent.
The history is kept in memory so the observer can compute metrics later.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from cognitiveweave.experiments.agent import CycleWrite, SocietyAgent

logger = logging.getLogger(__name__)


@dataclass
class CycleResult:
    cycle: int
    writes: list[CycleWrite]

    @property
    def successful_writes(self) -> list[CycleWrite]:
        return [w for w in self.writes if w.node_id is not None]


class AgentPopulation:
    """Manages a group of SocietyAgents and runs experiment cycles.

    Args:
        agents: The agents participating in this experiment.
    """

    def __init__(self, agents: list[SocietyAgent]) -> None:
        self.agents = agents
        self.history: list[CycleResult] = []

    # ------------------------------------------------------------------
    # Cycle execution
    # ------------------------------------------------------------------

    def run_cycle(self, cycle_num: int) -> CycleResult:
        """Run all agents for one cycle. Returns the combined result."""
        writes: list[CycleWrite] = []
        for agent in self.agents:
            try:
                write = agent.cycle(cycle_num)
                writes.append(write)
                logger.info(
                    "Population cycle %d: %s wrote %s",
                    cycle_num, agent.name, write.node_id or "(nothing)",
                )
            except Exception:
                logger.exception(
                    "Population cycle %d: agent %s crashed", cycle_num, agent.name
                )
        result = CycleResult(cycle=cycle_num, writes=writes)
        self.history.append(result)
        return result

    def run(self, n_cycles: int, *, start_cycle: int = 0) -> list[CycleResult]:
        """Run n_cycles sequentially. Returns all results."""
        results = []
        for i in range(start_cycle, start_cycle + n_cycles):
            results.append(self.run_cycle(i))
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def current_cycle(self) -> int:
        return len(self.history)

    @property
    def agent_names(self) -> list[str]:
        return [a.name for a in self.agents]
