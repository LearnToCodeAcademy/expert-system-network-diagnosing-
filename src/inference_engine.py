"""Forward-chaining inference engine for rule-based diagnosis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from src.rules import Rule, RuleKnowledgeBase


@dataclass
class RuleResult:
    diagnosis: str
    confidence: float
    recommendation: str
    source: str
    explanation: str
    fired_rules: List[str]


class ForwardChainingEngine:
    """Simple forward chaining engine with rule priority handling."""

    def __init__(self, knowledge_base: RuleKnowledgeBase):
        self.knowledge_base = knowledge_base

    def infer(self, symptoms: Dict[str, Any]) -> RuleResult:
        fired: List[Rule] = []

        for rule in self.knowledge_base.get_rules():
            if rule.evaluate(symptoms):
                fired.append(rule)

        if not fired:
            return RuleResult(
                diagnosis="unknown",
                confidence=0.0,
                recommendation="Insufficient rule match; defer to ML prediction.",
                source="rule-based",
                explanation="No rules were triggered by the provided symptoms.",
                fired_rules=[],
            )

        fired_sorted = sorted(fired, key=lambda r: (r.priority, r.confidence), reverse=True)
        best = fired_sorted[0]
        explanation = self._build_explanation(fired_sorted)

        return RuleResult(
            diagnosis=best.diagnosis,
            confidence=best.confidence,
            recommendation=best.recommendation,
            source="rule-based",
            explanation=explanation,
            fired_rules=[r.name for r in fired_sorted],
        )

    @staticmethod
    def _build_explanation(fired: List[Rule]) -> str:
        lines = [
            "Triggered rules (sorted by priority/confidence):",
        ]
        for rule in fired:
            lines.append(
                f"- {rule.name}: diagnosis={rule.diagnosis}, priority={rule.priority}, confidence={rule.confidence:.2f}"
            )
        return "\n".join(lines)
