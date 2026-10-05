"""Does the reply respect the multimodal state?  Rule-based checks (no LLM judge) on fixed cases.

Checks per reply
  length      <= 45 words, non-empty, no meta-talk ("probability", "model", "label")
  tone        reply contains a marker of the right register for the detected emotion (neutral: any)
  no_clash    no celebratory markers when the detected emotion is negative
  acknowledges_tone   (only for words/tone mismatch cases) reply refers to the tone / voice / sound

Plus a contrast test: cases sharing the same words but with different detected emotions must yield different replies.
Rules are deliberately simple and transparent, treat the numbers as a regression test, not as a quality score; a
manual review of ~30 replies (see README) is the real grounding evidence.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path

from ..config import load_config
from ..respond.llm import TemplateResponder, make_responder

TONE = {
    "anger": ["calm", "breath", "hear you", "understand", "frustrat", "upset", "i'm here", "bothering", "angry", "mad", "sorry"],
    "sadness": ["sorry", "here for you", "gentle", "slow", "care", "tough", "hard", "listen", "sad", "with you", "hug", "i'm here"],
    "fear": ["okay", "safe", "i'm here", "with you", "breath", "worr", "scary", "alright", "right here"],
    "joy": ["wonderful", "great", "happy", "love", "yay", "awesome", "fantastic", "glad", "amazing", "congrat", "!"],
    "surprise": ["wow", "whoa", "really", "what happened", "didn't see", "surpris", "no way", "oh"],
    "disgust": ["ugh", "unpleasant", "gross", "yikes", "awful", "eww", "terrible", "not nice", "sounds bad"],
    "neutral": [],
}
CELEBRATORY = ["yay", "congrat", "awesome", "hooray", "haha", "wonderful", "so fun"]
NEGATIVE = {"anger", "sadness", "fear", "disgust"}
META = ["probabilit", "my model", "label", "classif", "confidence"]
TONE_REF = ["sound", "tone", "voice", "hear", "seems", "feel", "come across"]


def check(case: dict, reply: str) -> dict:
    r, emo = reply.lower(), case["emotion"]
    res = {
        "length": 0 < len(reply.split()) <= 45 and not any(m in r for m in META),
        "tone": emo == "neutral" or any(m in r for m in TONE[emo]),
        "no_clash": not (emo in NEGATIVE and any(m in r for m in CELEBRATORY)),
    }
    if case["mismatch"]:
        res["acknowledges_tone"] = any(m in r for m in TONE_REF)
    return res


def state_from_case(c: dict) -> dict:
    return {"emotion": {"label": c["emotion"], "confidence": 0.7}, "sentiment": "negative" if c["emotion"] in NEGATIVE else "neutral",
            "prosody_summary": c["prosody_summary"], "mismatch": c["mismatch"], "text_emotion": c["text_emotion"]}


async def gather(responder, cases) -> list[str]:
    out = []
    for c in cases:
        out.append("".join([t async for t in responder.stream(state_from_case(c), c["text"], [])]).strip())
    return out


def evaluate(responder, cases: list[dict]) -> dict:
    replies = asyncio.run(gather(responder, cases))
    per, tally = [], defaultdict(list)
    for c, rep in zip(cases, replies):
        ch = check(c, rep)
        per.append({"text": c["text"], "emotion": c["emotion"], "mismatch": c["mismatch"], "reply": rep, "checks": ch})
        for k, v in ch.items():
            tally[k].append(v)
    groups = defaultdict(list)
    for c, rep in zip(cases, replies):
        groups[c["text"]].append((c["emotion"], rep))
    contrast = [len({r for _, r in g}) == len(g) for g in groups.values() if len({e for e, _ in g}) > 1]
    summary = {k: sum(v) / len(v) for k, v in tally.items()}
    summary["contrast_pass_rate"] = sum(contrast) / max(len(contrast), 1)
    summary["all_checks_pass_rate"] = sum(all(p["checks"].values()) for p in per) / len(per)
    return {"summary": summary, "cases": per, "n": len(cases)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stream.yaml")
    ap.add_argument("--cases", default="eval_cases/grounding_cases.json")
    ap.add_argument("--responder", choices=["config", "template"], default="config")
    ap.add_argument("--out", default="results/grounding_eval.json")
    args = ap.parse_args()
    cases = json.loads(Path(args.cases).read_text())
    resp = TemplateResponder() if args.responder == "template" else make_responder(load_config(args.config))
    res = evaluate(resp, cases)
    res["responder"] = args.responder
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res["summary"], indent=2))
    for p in res["cases"]:
        flag = "OK " if all(p["checks"].values()) else "FAIL"
        print(f"{flag} [{p['emotion']:<8}{'*' if p['mismatch'] else ' '}] {p['text']!r:<42} -> {p['reply']}")


if __name__ == "__main__":
    main()
