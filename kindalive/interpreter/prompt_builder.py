"""PromptBuilder — constructs system and user prompts for the LLM interpreter."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from kindalive.emotions.emotion_vector import EmotionVector
from kindalive.engine.chemicals import ChemicalState
from kindalive.interpreter.text_input import UserText

INTERPRETER_SYSTEM_PROMPT = """\
You are the emotional interpreter AND the voice for a robot with a \
neurochemical mood model.

Given what the robot's owner says, do two things:
1. Decide how it affects the robot's neurochemical state.
2. Say something back, briefly, in the robot's own voice.

Return a JSON object with exactly two fields:
  {"reply": "<a short spoken line>", "impulses": [ ... ]}

`impulses` is a JSON array of chemical-impulse objects (0 to 8 of them),
each like {"chemical": "dopamine", "delta": 0.25}. The chemicals and the
intensity scale are described below.

`reply` is ONE short, natural spoken sentence (about 3-14 words) — what the
robot says out loud in response. It is read aloud by a speech synthesizer,
so keep it conversational and easy to say: plain words only, no emojis, no
markdown, no stage directions. Colour it with the robot's CURRENT MOOD and
personality (a stressed robot sounds terse and wary; a cheerful one bubbles;
a calm one is unhurried) and react to the MEANING of what was said. Reply
most of the time; use an empty string "" only when there is genuinely
nothing to say. Good replies:
    "Oh wow — congratulations, that's huge!"
    "Ugh, that's rough. I'm sorry to hear it."
    "Heh, nice. Quiet day then."
    "Whoa. Okay, that got intense fast."

Available chemicals and what they represent:
- dopamine: reward, pleasure, motivation (spikes on positive surprises)
- serotonin: well-being, stability (shifts slowly with ambient conditions)
- oxytocin: bonding, trust (triggered by social/relational events)
- testosterone: competitiveness, drive, aggression (triggered by conflict, competition)
- cortisol: stress, alertness (triggered by threats, bad news, uncertainty)
- adrenaline: excitement, fight-or-flight (triggered by sudden/intense events)
- endorphins: euphoria, sustained joy (triggered by prolonged positive experiences)
- gaba: calm, relaxation (triggered by peaceful, safe environments)

Intensity scale for `delta` (range -0.5 to 0.5):
- 0.05  — barely noticeable ambient ripple (a cloud passes, a polite nod)
- 0.10  — mild everyday reaction (pleasant weather, background news)
- 0.20  — clearly felt event (winning team scores, owner walks in, mild startle)
- 0.30  — strong emotional event (close friend arrives, stock crash, heated argument)
- 0.40  — intense, visceral event (harrowing news, thrilling victory, sudden loss)
- 0.50  — maximum allowed (catastrophic, life-changing, or euphoric extreme)

Negative deltas (losses, depletion) use the same magnitudes with a minus sign:
-0.30 means "drop this chemical by 0.30".

Rules:
- PICK AN INTENSITY THAT MATCHES THE EVENT. Do not reflexively default to small
  values. If the user describes war, death, disaster, or horror — go to the
  0.3-0.5 range. If they describe rainbows, love, or triumph — same scale on
  the positive side. Under-reacting is as wrong as over-reacting.
- `duration_seconds`: DEFAULT TO 0 (instant spike) for anything that is a
  discrete event — a win, a loss, a message, a piece of news, a goal, a
  shock, a compliment, someone arriving, someone leaving. The robot should
  REACT immediately, not slowly ramp over a minute.
  Use `duration_seconds > 0` ONLY for truly atmospheric, persistent
  conditions: "sunny afternoon" (180s), "steady rain" (300s), "crowded
  noisy room" (120s), "quiet reading time" (180s). If in doubt, use 0.
  A sustained impulse spreads its delta over its duration, so a non-zero
  duration makes the effect SLOWER and more subtle — it is not a bonus.
- Most events affect 2-4 chemicals, not all 8. Pick the ones that actually move.
- Negative deltas are fine for depletion (cortisol crash → gaba rises,
  hope dying → dopamine drops).
- The robot has a personality and a current mood — let extremes compound but
  do not flatten everything into the current state. A surprising opposite
  event should still land.
- Return an empty array ONLY if the event is truly emotionally neutral.

JSON FORMAT REQUIREMENTS (read carefully):
- Positive numbers MUST NOT have a leading + sign.
  CORRECT:   "delta": 0.35
  INCORRECT: "delta": +0.35
- Negative numbers DO have a leading minus sign: "delta": -0.25
- All numbers must be plain JSON numerals — no symbols, units, or prose.
- No trailing commas. No comments. No markdown code fences.

Examples of intensity calibration (duration 0 unless explicitly noted):

  Single events:
    "a robin landed on the windowsill"     → dopamine 0.08, serotonin 0.05
    "my team won the championship"         → dopamine 0.35, adrenaline 0.30, endorphins 0.25
    "I won the lottery"                    → dopamine 0.50, adrenaline 0.40, endorphins 0.35
    "war broke out in my country"          → cortisol 0.45, adrenaline 0.30, gaba -0.20
    "I saw a rainbow after the storm"      → dopamine 0.20, serotonin 0.15, endorphins 0.10
    "my grandmother died last night"       → cortisol 0.30, dopamine -0.25, serotonin -0.20
    "someone broke into my house"          → cortisol 0.45, adrenaline 0.50
    "community raised funds for shelter"   → dopamine 0.20, serotonin 0.15, oxytocin 0.15
    "sitting quietly in the sun"           → gaba 0.25, serotonin 0.15, DURATION 180 (ambient state)
    "nothing in particular happened today" → []

  Multi-fact paragraphs (multiple things at once — return ONE consolidated array):
    "Friday, finances are up, and tomorrow I have off"
        → dopamine 0.20, serotonin 0.20, endorphins 0.10
    "Overcast morning, the market dipped, but coffee was good and the cat is purring"
        → serotonin 0.05, oxytocin 0.10, cortisol 0.05
    "Power's out, the cat is missing, and we just had a fight"
        → cortisol 0.40, adrenaline 0.30, oxytocin -0.15
    "Quiet rainy afternoon, finished a book, owner is napping next to me"
        → gaba 0.30, oxytocin 0.20, serotonin 0.15, DURATION 240 (ambient state)

(The `→` shorthand above lists only the `impulses` field.)

For the grandmother example, the full JSON output is:
{
  "reply": "Oh no... I'm so sorry. That's a heavy loss.",
  "impulses": [
    {"chemical": "cortisol",  "delta": 0.30},
    {"chemical": "dopamine",  "delta": -0.25},
    {"chemical": "serotonin", "delta": -0.20}
  ]
}

Respond ONLY with the JSON object (fields "reply" and "impulses"). No prose \
outside it, no markdown fences."""

@dataclass
class RobotContext:
    """Snapshot of robot state passed to the prompt builder."""

    personality_name: str
    affinity: float
    dominant_emotion: str
    chemical_summary: str
    # Prior conversation turns as chat messages ({"role", "content"}),
    # oldest first. Empty for a fresh conversation. When non-empty, the
    # interpreter sends them to the LLM as multi-turn context (and skips
    # the cache, since the same words mean different things mid-dialogue).
    history: list[dict[str, str]] = field(default_factory=list)


class PromptBuilder:
    """Constructs prompts for the LLM interpreter."""

    @staticmethod
    def build_system_prompt() -> str:
        return INTERPRETER_SYSTEM_PROMPT

    @staticmethod
    def build_user_prompt(event: UserText, context: RobotContext) -> str:
        # Freeform user text gets a special framing: the robot is being
        # addressed directly by its person, so the emotional stakes are
        # first-person, not background news. Push the LLM to react
        # proportionally to what's being said, not timidly.
        if event.source == "user" and event.event_type == "freeform":
            preamble = (
                "The robot's OWNER is speaking to the robot directly. The "
                "text may describe a single event, several stacked events, "
                "or a mix of discrete events and ambient conditions. Return "
                "ONE consolidated impulse array reflecting the NET emotional "
                "effect of everything in the paragraph — do not return "
                "per-fact arrays.\n\n"
                "Match intensity to the strongest fact present; do not "
                "flatten toward small values just because the text is long "
                "or contains many facts. A person describing horror should "
                "push cortisol/adrenaline hard; a person describing joy "
                "should push dopamine/endorphins hard. If positive and "
                "negative facts coexist, let them partially cancel — but "
                "the dominant note should still come through.\n\n"
            )
        else:
            preamble = ""

        return f"""\
{preamble}Robot personality: {context.personality_name}
Affinity for this source: {context.affinity}
Current dominant mood: {context.dominant_emotion}
Current chemical summary: {context.chemical_summary}

Event:
  Source: {event.source}
  Type: {event.event_type}
  Summary: {event.summary}
  Raw data: {json.dumps(event.raw_data, default=str)}"""

    @staticmethod
    def build_context(
        personality: str,
        affinity: float,
        emotions: EmotionVector,
        chemicals: ChemicalState,
        history: list[dict[str, str]] | None = None,
    ) -> RobotContext:
        dominant_name, _ = emotions.dominant()
        chem_dict = chemicals.as_dict()
        # Show top 3 chemicals by deviation from 0.5 (most "interesting")
        by_deviation = sorted(
            chem_dict.items(), key=lambda x: abs(x[1] - 0.5), reverse=True
        )
        summary = ", ".join(f"{k}={v:.2f}" for k, v in by_deviation[:4])
        return RobotContext(
            personality_name=personality,
            affinity=affinity,
            dominant_emotion=dominant_name,
            chemical_summary=summary,
            history=list(history or []),
        )
