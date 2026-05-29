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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from cognitiveweave.experiments.agent import CycleWrite, SocietyAgent

if TYPE_CHECKING:
    from cognitiveweave.experiments.epistemic_monitor import EpistemicMonitor

logger = logging.getLogger(__name__)


@dataclass
class CycleResult:
    cycle: int
    writes: list[CycleWrite]
    entropy: float | None = field(default=None)   # system entropy this cycle
    regime: str | None = field(default=None)

    @property
    def successful_writes(self) -> list[CycleWrite]:
        return [w for w in self.writes if w.node_id is not None]


class AgentPopulation:
    """Manages a group of SocietyAgents and runs experiment cycles.

    Args:
        agents:  The agents participating in this experiment.
        monitor: Optional EpistemicMonitor — computes entropy after each cycle
                 and writes it to the graph so agents can adapt next cycle.
    """

    def __init__(
        self,
        agents: list[SocietyAgent],
        *,
        monitor: EpistemicMonitor | None = None,
    ) -> None:
        self.agents = agents
        self.monitor = monitor
        self.history: list[CycleResult] = []

    # ------------------------------------------------------------------
    # Cycle execution
    # ------------------------------------------------------------------

    def run_cycle(self, cycle_num: int) -> CycleResult:
        """Run all agents for one cycle, then update epistemic state."""
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

        # After all agents write, compute entropy and publish to graph
        # so agents can adapt their behavior in the NEXT cycle.
        if self.monitor is not None:
            successful = result.successful_writes
            if successful:
                beliefs = {w.agent: w.belief for w in successful if w.belief}
                faiss = next(
                    (a._faiss for a in self.agents if a._faiss is not None), None
                )
                if beliefs and faiss is not None:
                    from cognitiveweave.telemetry import system_entropy as entropy_metric
                    state = self.monitor.observe(cycle_num, beliefs, faiss.encode)
                    entropy_metric.record(state.entropy, {"regime": state.regime})
                    result.entropy = state.entropy
                    result.regime = state.regime

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
