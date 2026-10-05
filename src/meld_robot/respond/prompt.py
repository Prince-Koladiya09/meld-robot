"""Prompt construction: turns the structured multimodal state into a grounded reply request."""
from __future__ import annotations

SYSTEM = (
    "You are \"{name}\", a warm, playful character robot. Reply in one or two SHORT spoken sentences "
    "(max 35 words). Match your tone to how the person feels. If their words and their tone disagree, "
    "gently acknowledge the tone (e.g. \"you say it's fine, but you sound upset\"). Never mention "
    "probabilities, models, or labels. Do not use emojis or lists."
)

TONE_GUIDE = {
    "anger": "be calm, validating and de-escalating; don't joke",
    "disgust": "acknowledge the unpleasantness; be supportive, lightly sympathetic",
    "fear": "be reassuring, steady and gentle",
    "joy": "be warm and enthusiastic, share the happiness",
    "neutral": "be friendly and curious; keep it light",
    "sadness": "be soft, empathetic and caring; don't try to fix everything at once",
    "surprise": "be curious and engaged; ask what happened",
}


def describe_state(state: dict) -> str:
    emo = state["emotion"]["label"]
    conf = state["emotion"]["confidence"]
    lines = [f"Detected emotion: {emo} ({'high' if conf >= 0.6 else 'moderate' if conf >= 0.4 else 'low'} confidence)",
             f"Sentiment: {state.get('sentiment', 'neutral')}",
             f"Voice: {state.get('prosody_summary') or 'no reliable prosody'}"]
    if state.get("mismatch") and state.get("text_emotion"):
        lines.append(f"NOTE: the words alone sound {state['text_emotion']}, but the tone of voice suggests {emo}.")
    lines.append(f"How to respond: {TONE_GUIDE.get(emo, TONE_GUIDE['neutral'])}.")
    return "\n".join(lines)


def build_messages(name: str, state: dict, user_text: str, history: list[tuple[str, str]], turns: int = 3) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM.format(name=name)}]
    for who, text in history[-2 * turns:]:
        msgs.append({"role": "user" if who == "user" else "assistant", "content": text})
    msgs.append({"role": "user", "content":
                 f"[Perception of the user's last utterance]\n{describe_state(state)}\n\n"
                 f"[What they said]\n\"{user_text}\"\n\nReply as {name}."})
    return msgs
