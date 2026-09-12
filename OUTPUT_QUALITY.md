# Output quality contract

Scope: preserve the two vocal personas, command paths, public signatures and existing JSON keys. Improve artifact fidelity without changing models or installing dependencies. Source files take precedence over the architecture diagram.

| Module | Responsibility | Depends on |
|---|---|---|
| vocal-blueprint | Preserve user lyrics, genre, production and actionable steering | input interpreter, vocal bible |
| vocal-reference | Deliver the separated vocal and distinct reference excerpts | separation, ranking |
| melodic-guide | Render valid MIDI, preserve note timing, avoid accompaniment duplication | transcription |
| analysis-export | Publish final embeddings and measured trajectories with provenance | acoustic analysis, embedding |

Build order: vocal-blueprint → vocal-reference → melodic-guide → analysis-export. Each increment gets focused unittest coverage and compilation checks; real neural/audio quality requires the deployment dependencies and representative recordings.

Acceptance checks:
- Blueprint: retain repeated sections in original order and preserve lyrics/case; honor genre and production; steering changes copyable text; enforce 1000/4000 character budgets without silently truncating lyrics; report validation failures honestly.
- Reference: never label the original mix as isolated vocals; check subprocess failures and required outputs; preserve the unprocessed separated vocal separately from reference clips.
- Guide: accept MIDI based on playable notes, preserve their pitches and timing, and reject failed renders; avoid playing the complete accompaniment twice as melody and chords.
- Export: final vocal JSON contains the completed identity; raw embeddings remain full precision and identify the actual model; failed learned embeddings are never described as successful ECAPA output; pitch/voicing trajectories include timebase and explicit missing values.

Listening acceptance remains open: compare original and output on the same passages for bleed, consonant/transient loss, pitch errors, dropouts, timbral damage and clipping. Assess blueprints using controlled Suno generations. A passing unit test is not a perceptual quality score.

Suno v5.5 voice identity requires its audio/Voice workflow; text descriptors alone are not speaker verification. Official reference: https://help.suno.com/en/articles/11362369 (checked 2026-09-12).

Praat Integration: F1–F4 formants (Burg algorithm) and HNR (cross-correlation Harmonicity) are now wired directly into VocalDNAAnalyzer and mapped into Suno text prompts (Singer's Formant ring at 2.6–3.4kHz, pharyngeal twang vs cavernous throat, open belting vs covered acoustics, and HNR harmonic clarity/breath rasp). The user explicitly selected text-only Suno prompts over audio voice references.

Implemented: source-owned section ordering/lyrics; genre/production preservation; actionable diction/rhythm and mode tokens; honest prompt validation and box limits; checked separation failures; preserved isolated waveform; playable-note MIDI validation; separate melody delivery; removal of invented drum grid; final identity refresh; full-precision per-segment embeddings; raw pitch/voicing/MFCC trajectories and fallback provenance. Existing summary fields retain compatibility; raw JSON identifies unavailable measurements and legacy defaults.

Verification: focused unittest coverage with inference/audio I/O replaced only at external boundaries, plus Python compilation. Existing `test_suno_adapter_prompts_generation` fails on both HEAD and the working tree because it expects `5.2Hz natural vibrato`; it remains unchanged. No model inference, Suno generation, listening assessment or server deployment has been performed.

Runtime prerequisites are unavailable in this local Python. In addition to the listed requirements, source imports `audio_separator`, `speechbrain` and `parselmouth`; corresponding distributions and system tools/models need deployment verification. No dependencies were installed or configuration changed.
