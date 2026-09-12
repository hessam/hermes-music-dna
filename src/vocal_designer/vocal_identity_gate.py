"""Vocal Identity Gate: Evaluates Broadway front-load, explicit gender/register, and core anchors."""
from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from music_dna_agent.src.vocal_designer.vocal_bible import VocalBibleBase
from music_dna_agent.src.vocal_designer.input_interpreter import InterpretationResult

logger = logging.getLogger(__name__)


@dataclass
class IdentityAnchorEvaluation:
    anchor_name: str
    passed: bool
    score: float
    observed_trait: str
    feedback: str


@dataclass
class IdentityGateResult:
    passed: bool
    identity_distance: int
    anchors: List[IdentityAnchorEvaluation]
    repair_needed: bool
    repair_patch: Optional[Dict[str, Any]] = None
    summary: str = ""


class VocalIdentityGate:
    """Validates that Broadway override is #1 and explicit gender/register is front-loaded."""

    @classmethod
    def evaluate(
        cls,
        bible: VocalBibleBase,
        interpretation: InterpretationResult,
        compiled_bundle: Dict[str, Any],
    ) -> IdentityGateResult:
        style_prompt = (compiled_bundle.get("style_prompt") or "")
        style_lower = style_prompt.lower()
        anchors: List[IdentityAnchorEvaluation] = []
        penalties = 0

        # Anchor 1: Broadway Override at #1
        has_broadway_start = style_lower.startswith("broadway")
        if not has_broadway_start:
            penalties += 30
            anchors.append(IdentityAnchorEvaluation(
                anchor_name="Broadway Global Override",
                passed=False,
                score=0,
                observed_trait="Broadway is not at the start of Style Box.",
                feedback="Place 'Broadway' at the very beginning of the Style prompt.",
            ))
        else:
            anchors.append(IdentityAnchorEvaluation(
                anchor_name="Broadway Global Override",
                passed=True,
                score=100,
                observed_trait="Broadway front-loaded at token #1.",
                feedback="Global clarity override active.",
            ))

        # Anchor 2: Explicit Gender & Register in First 20 Words
        has_explicit_gender = bible.gender in style_lower[:60]
        if not has_explicit_gender:
            penalties += 25
            anchors.append(IdentityAnchorEvaluation(
                anchor_name="Explicit Gender & Register",
                passed=False,
                score=20,
                observed_trait=f"Missing explicit '{bible.gender}' front-load.",
                feedback=f"Specify '{bible.gender} {bible.identity_card.vocal_register}' in first 20 words.",
            ))
        else:
            anchors.append(IdentityAnchorEvaluation(
                anchor_name="Explicit Gender & Register",
                passed=True,
                score=100,
                observed_trait=f"Explicit {bible.gender} register front-loaded.",
                feedback="Tier-1 vocal control active.",
            ))

        # Anchor 3: 4-7 Sweet Spot Length
        tokens = [t.strip() for t in style_prompt.split(",") if t.strip()]
        is_sweet_spot = (4 <= len(tokens) <= 9)
        if not is_sweet_spot:
            penalties += 15
            anchors.append(IdentityAnchorEvaluation(
                anchor_name="4-7 Descriptor Sweet Spot",
                passed=False,
                score=50,
                observed_trait=f"Token count ({len(tokens)}) out of optimal range.",
                feedback="Keep descriptors within 4-7 sweet spot.",
            ))
        else:
            anchors.append(IdentityAnchorEvaluation(
                anchor_name="4-7 Descriptor Sweet Spot",
                passed=True,
                score=100,
                observed_trait=f"Optimal descriptor count ({len(tokens)} tokens).",
                feedback="High mathematical prompt priority.",
            ))

        identity_distance = min(100, penalties)
        passed = (identity_distance <= 30)
        repair_needed = not passed

        patch = None
        if repair_needed:
            patch = {"restore": ["Broadway", f"{bible.gender} {bible.identity_card.vocal_register}"]}

        return IdentityGateResult(
            passed=passed,
            identity_distance=identity_distance,
            anchors=anchors,
            repair_needed=repair_needed,
            repair_patch=patch,
            summary=f"Identity Distance: {identity_distance}/100 ({'PASSED' if passed else 'REPAIRED'})",
        )

    @classmethod
    def apply_repair(
        cls,
        compiled_bundle: Dict[str, Any],
        gate_result: IdentityGateResult,
    ) -> Dict[str, Any]:
        if not gate_result.repair_needed:
            return compiled_bundle

        style = compiled_bundle.get("style_prompt", "")
        tokens = [t.strip() for t in style.split(",") if t.strip()]
        if not style.lower().startswith("broadway"):
            tokens.insert(0, "Broadway")

        compiled_bundle["style_prompt"] = ", ".join(tokens) + "."
        return compiled_bundle
