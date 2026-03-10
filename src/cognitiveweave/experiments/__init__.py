"""Emergent Social Epistemology experiment layer.

A closed population of LLM agents share a CognitiveWeave knowledge graph.
Each agent has asymmetric initial knowledge and limited communication bandwidth.
We observe how shared beliefs form, drift, and stabilize over time.
"""
from cognitiveweave.experiments.agent import SocietyAgent
from cognitiveweave.experiments.population import AgentPopulation
from cognitiveweave.experiments.observer import ExperimentObserver

__all__ = ["SocietyAgent", "AgentPopulation", "ExperimentObserver"]
