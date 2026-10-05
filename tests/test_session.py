import asyncio

import numpy as np

from meld_robot.config import load_config
from meld_robot.labels import EMOTIONS
from meld_robot.respond.llm import TemplateResponder
from meld_robot.stream.engine import EngineOutput
from meld_robot.stream.pipeline import Components, Session
from meld_robot.stream.vad import EnergyVAD

CFG = load_config("configs/stream.yaml")
SR = 16000


class FakeASR:
    def transcribe(self, audio, final=False):
        return "no it's fine" if len(audio) > 4000 else ""


class FakeEngine:
    """Text says neutral, fused/audio says anger -> mismatch must be flagged."""
    def predict(self, text, context, audio):
        p = np.full(7, 0.02); p[EMOTIONS.index("anger")] = 0.88
        t = np.full(7, 0.02); t[EMOTIONS.index("neutral")] = 0.88
        return EngineOutput(p / p.sum(), t / t.sum(), p / p.sum(), 0.3)


def speech(seconds, amp=0.3):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * 180 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)


def silence(seconds):
    return np.zeros(int(seconds * SR), np.float32)


def make(responder=None, realtime=False):
    comp = Components(EnergyVAD(-45), FakeASR(), FakeEngine(), responder or TemplateResponder())
    q: asyncio.Queue = asyncio.Queue()
    return Session(CFG, comp, q, realtime=realtime), q


def drain(q):
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


async def test_one_utterance_end_to_end():
    s, q = make()
    audio = np.concatenate([silence(0.4), speech(1.5), silence(1.0)])
    for i in range(0, len(audio), 320):          # 20 ms frames, as the client sends them
        await s.feed((audio[i:i + 320] * 32767).astype(np.int16).tobytes())
    await s.flush()
    ev = drain(q)
    types = [e["type"] for e in ev]

    assert types.index("turn") == 0 and ev[0]["turn"] == "listening"
    assert "state_update" in types
    assert types.index("state_update") < types.index("state_final") < types.index("reply_token") < types.index("reply_done")
    assert ev[-1] == {**ev[-1], "type": "turn", "turn": "idle"}

    final = next(e for e in ev if e["type"] == "state_final")
    assert final["text"] == "no it's fine"
    assert final["emotion"]["label"] == "anger" and final["text_emotion"] == "neutral"
    assert final["mismatch"] is True and final["sentiment"] == "negative"
    assert "prosody" in final and final["latency_ms"]["asr"] >= 0

    upd = [e for e in ev if e["type"] == "state_update"]
    assert len(upd) >= 3 and all(abs(sum(e["emotion"]["probs"].values()) - 1) < 0.05 for e in upd)
    reply = next(e for e in ev if e["type"] == "reply_done")["text"]
    assert "sound angry" in reply                # mismatch phrasing reached the response
    assert s.history[0] == ("user", "no it's fine") and s.history[1][0] == "robot"


async def test_silence_only_produces_no_turn():
    s, q = make()
    await s.feed(silence(2.0))
    await s.flush()
    assert drain(q) == []


async def test_barge_in_cancels_reply():
    s, q = make(TemplateResponder(delay_s=0.05), realtime=True)

    async def play(a):
        for i in range(0, len(a), 320):
            await s.feed(a[i:i + 320])
            await asyncio.sleep(0.002)

    await play(np.concatenate([speech(1.2), silence(0.8)]))
    await asyncio.sleep(0.6)                      # reply is mid-stream (slow responder)
    mid = drain(q)
    assert any(e["type"] == "reply_token" for e in mid) and not any(e["type"] == "reply_done" for e in mid)

    await play(np.concatenate([speech(1.0), silence(0.8)]))   # user interrupts
    await s.flush()
    ev = drain(q)
    types = [e["type"] for e in ev]
    assert "barge_in" in types and "reply_cancelled" in types
    assert types.count("state_final") == 1        # the 2nd utterance's final state
    assert ev[[e["type"] for e in ev].index("barge_in")]["type"] == "barge_in"
    await s.close()


def test_prompt_contains_state_and_mismatch_note():
    from meld_robot.respond.prompt import build_messages
    st = {"emotion": {"label": "anger", "confidence": 0.58}, "sentiment": "negative",
          "prosody_summary": "energy high, speaking rate fast", "mismatch": True, "text_emotion": "neutral"}
    m = build_messages("Pip", st, "No, it's fine.", [("user", "hi"), ("robot", "hello!")])
    assert m[0]["role"] == "system" and m[1]["content"] == "hi" and m[2]["role"] == "assistant"
    last = m[-1]["content"]
    assert "anger" in last and "words alone sound neutral" in last and "energy high" in last and "No, it's fine." in last
    assert "0.58" not in last                      # raw probabilities must not leak into the prompt


def test_grounding_eval_template_baseline():
    import json
    from meld_robot.eval.grounding_eval import evaluate
    cases = json.load(open("eval_cases/grounding_cases.json"))
    res = evaluate(TemplateResponder(), cases)
    s = res["summary"]
    assert s["length"] == s["tone"] == s["no_clash"] == s["acknowledges_tone"] == 1.0
    assert s["contrast_pass_rate"] == 1.0
