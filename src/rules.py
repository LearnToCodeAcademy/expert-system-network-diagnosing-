"""Knowledge base definitions for the network troubleshooting expert system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


Condition = Tuple[str, str, float]


@dataclass(frozen=True)
class Rule:
    """A single IF-THEN rule in the knowledge base."""

    name: str
    conditions: List[Condition]
    diagnosis: str
    recommendation: str
    priority: int = 1
    confidence: float = 0.8
    description: Optional[str] = None

    def evaluate(self, facts: Dict[str, Any]) -> bool:
        """Return True when all conditions are met by input facts."""
        for feature, operator, value in self.conditions:
            if feature not in facts:
                return False
            fact_value = float(facts[feature])
            if operator == ">" and not (fact_value > value):
                return False
            if operator == ">=" and not (fact_value >= value):
                return False
            if operator == "<" and not (fact_value < value):
                return False
            if operator == "<=" and not (fact_value <= value):
                return False
            if operator == "==" and not (fact_value == value):
                return False
        return True


@dataclass
class RuleKnowledgeBase:
    """Container for all network diagnostics rules."""

    rules: List[Rule] = field(default_factory=list)

    @classmethod
    def default(cls) -> "RuleKnowledgeBase":
        """Build a default rule set for network troubleshooting."""
        default_rules = [
            Rule(
                name="critical_packet_loss",
                conditions=[("packet_loss", ">", 5.0)],
                diagnosis="packet_loss_issue",
                recommendation="Check physical links, replace faulty cables, and inspect WAN quality.",
                priority=10,
                confidence=0.95,
                description="High packet loss immediately indicates delivery issues.",
            ),
            Rule(
                name="high_congestion",
                conditions=[("congestion", ">", 0.7), ("bandwidth_usage", ">", 80.0)],
                diagnosis="congestion_issue",
                recommendation="Enable QoS, shape traffic, or add capacity during peak times.",
                priority=9,
                confidence=0.9,
                description="Sustained congestion with heavy utilization implies bottlenecks.",
            ),
            Rule(
                name="slow_network_latency",
                conditions=[("latency", ">", 80.0), ("packet_loss", "<", 2.0), ("jitter", ">", 15.0)],
                diagnosis="slow_network",
                recommendation="Investigate routing path, DNS delays, and overloaded transit links.",
                priority=8,
                confidence=0.88,
                description="High delay with low loss often points to slowness, not drops.",
            ),
            Rule(
                name="traffic_anomaly_attack",
                conditions=[("traffic_anomaly", ">", 0.75), ("packet_loss", ">=", 1.5)],
                diagnosis="security_attack",
                recommendation="Activate DDoS mitigation and inspect firewall/IDS alerts.",
                priority=11,
                confidence=0.97,
                description="Anomalous traffic spikes with degradation can indicate attacks.",
            ),
            Rule(
                name="healthy_network",
                conditions=[("latency", "<=", 50.0), ("packet_loss", "<=", 1.0), ("congestion", "<=", 0.5)],
                diagnosis="normal",
                recommendation="No action needed; continue monitoring baseline metrics.",
                priority=1,
                confidence=0.85,
                description="Metrics in normal range indicate healthy network behavior.",
            ),
        ]
        return cls(rules=default_rules)

    def add_rule(self, rule: Rule) -> None:
        self.rules.append(rule)

    def get_rules(self) -> List[Rule]:
        return list(self.rules)
