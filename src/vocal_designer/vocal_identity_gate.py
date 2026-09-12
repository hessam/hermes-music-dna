"""Validate textual prompt constraints; this is not acoustic speaker verification."""
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
    """Validate requested genre, explicit voice register and the style budget."""

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

        # Preserve the requested song world independently of singer identity.
        has_genre = interpretation.hard_constraints.genre.casefold() in style_lower
        if not has_genre:
            penalties += 30
            anchors.append(IdentityAnchorEvaluation(
                anchor_name="Requested genre",
                passed=False,
                score=0,
                observed_trait="Requested genre missing from Style Box.",
                feedback="Restore the requested genre without replacing it with a global style.",
            ))
        else:
            anchors.append(IdentityAnchorEvaluation(
                anchor_name="Requested genre",
                passed=True,
                score=100,
                observed_trait="Requested genre retained.",
                feedback="Song world preserved.",
            ))

        # Anchor 2: Explicit Gender & Register in First 20 Words
        has_explicit_gender = f"{bible.gender} {bible.identity_card.vocal_register}" in style_lower[:100]
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
        is_sweet_spot = bool(style_prompt.strip()) and len(style_prompt) <= 1000
        if not is_sweet_spot:
            penalties += 15
            anchors.append(IdentityAnchorEvaluation(
                anchor_name="Style box budget",
                passed=False,
                score=50,
                observed_trait=f"Style length ({len(style_prompt)}) exceeds its budget or is empty.",
                feedback="Keep the style box within 1000 characters.",
            ))
        else:
            anchors.append(IdentityAnchorEvaluation(
                anchor_name="Style box budget",
                passed=True,
                score=100,
                observed_trait=f"Style length: {len(style_prompt)} characters.",
                feedback="Style fits the copyable box.",
            ))

        identity_distance = min(100, penalties)
        passed = all(anchor.passed for anchor in anchors)
        repair_needed = not passed

        patch = None
        if repair_needed:
            patch = {"restore": [f"{bible.gender} {bible.identity_card.vocal_register}", interpretation.hard_constraints.genre]}

        return IdentityGateResult(
            passed=passed,
            identity_distance=identity_distance,
            anchors=anchors,
            repair_needed=repair_needed,
            repair_patch=patch,
            summary=f"Identity Distance: {identity_distance}/100 ({'PASSED' if passed else 'FAILED'})",
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
        for token in reversed((gate_result.repair_patch or {}).get("restore", [])):
            if token.casefold() not in style.casefold():
                tokens.insert(0, token)

        compiled_bundle["style_prompt"] = ", ".join(tokens) + "."
        return compiled_bundle
