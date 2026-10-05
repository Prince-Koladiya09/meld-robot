"""Response generators. Both expose:  async stream(state, user_text, history) -> async iterator of text pieces."""
from __future__ import annotations

import asyncio
import threading
from typing import AsyncIterator

from .prompt import build_messages


class TemplateResponder:
    """No-LLM fallback (CPU-only machines, tests, and the baseline for grounding evaluation)."""

    LINES = {
        "anger": "I hear you, and I'm not going anywhere. Take a breath with me. What's bothering you most?",
        "disgust": "Ugh, that does sound unpleasant. Do you want to talk it through?",
        "fear": "Hey, it's okay, I'm right here with you. What's worrying you?",
        "joy": "That's wonderful, I love hearing you sound so happy! Tell me more!",
        "neutral": "Got it! I'm listening. What would you like to do next?",
        "sadness": "Oh, I'm sorry. I'm here for you, and we can take it slowly. Want to tell me about it?",
        "surprise": "Whoa, really? I didn't see that coming! What happened?",
    }
    ADJ = {"anger": "angry", "disgust": "disgusted", "fear": "scared", "joy": "happy", "neutral": "calm",
           "sadness": "sad", "surprise": "surprised"}
    MISMATCH = "That didn't quite sound like what your words said, you sound {emo} to me. "

    def __init__(self, name: str = "Pip", delay_s: float = 0.0):
        self.name, self.delay = name, delay_s

    def param_count(self) -> int:
        return 0

    async def stream(self, state: dict, user_text: str, history) -> AsyncIterator[str]:
        emo = state["emotion"]["label"]
        text = (self.MISMATCH.format(emo=self.ADJ.get(emo, emo)) if state.get("mismatch") else "") + self.LINES.get(emo, self.LINES["neutral"])
        for w in text.split(" "):
            if self.delay:
                await asyncio.sleep(self.delay)
            yield w + " "


class TransformersResponder:
    """Local small instruct LLM with token streaming. Generation runs in a worker thread."""

    def __init__(self, model: str, device: str = "auto", quantization: str = "none", max_new_tokens: int = 60,
                 temperature: float = 0.7, top_p: float = 0.9, name: str = "Pip", history_turns: int = 3):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from ..config import pick_device

        self.torch, self.name, self.turns, self.model_name = torch, name, history_turns, model
        self.max_new, self.temp, self.top_p = max_new_tokens, temperature, top_p
        dev = pick_device(device)
        self.tok = AutoTokenizer.from_pretrained(model)
        kw: dict = {}
        if quantization in ("4bit", "8bit") and dev == "cuda":
            from transformers import BitsAndBytesConfig

            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=quantization == "4bit", load_in_8bit=quantization == "8bit",
                bnb_4bit_compute_dtype=torch.bfloat16)
            kw["device_map"] = {"": 0}
        else:
            kw["torch_dtype"] = torch.bfloat16 if dev == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model, **kw)
        if "device_map" not in kw:
            self.model.to(dev)
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def param_count(self) -> int:
        """Full logical parameter count from the architecture (4-bit packing would under-report real tensors)."""
        from transformers import AutoConfig, AutoModelForCausalLM

        with self.torch.device("meta"):
            m = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(self.model_name))
        return sum(p.numel() for p in m.parameters())

    def _generate(self, messages, loop, queue, stop: threading.Event) -> None:
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        class Stop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kw):
                return stop.is_set()

        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)

        def run():
            try:
                with self.torch.no_grad():
                    self.model.generate(ids, attention_mask=self.torch.ones_like(ids), max_new_tokens=self.max_new,
                                        do_sample=self.temp > 0, temperature=max(self.temp, 1e-4), top_p=self.top_p,
                                        streamer=streamer, stopping_criteria=StoppingCriteriaList([Stop()]),
                                        pad_token_id=self.tok.eos_token_id)
            finally:
                streamer.end()

        t = threading.Thread(target=run, daemon=True)
        t.start()
        try:
            for piece in streamer:
                loop.call_soon_threadsafe(queue.put_nowait, piece)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    async def stream(self, state: dict, user_text: str, history) -> AsyncIterator[str]:
        messages = build_messages(self.name, state, user_text, history, self.turns)
        loop, queue, stop = asyncio.get_running_loop(), asyncio.Queue(), threading.Event()
        worker = threading.Thread(target=self._generate, args=(messages, loop, queue, stop), daemon=True)
        worker.start()
        try:
            while True:
                piece = await queue.get()
                if piece is None:
                    break
                yield piece
        finally:
            stop.set()  # barge-in / cancellation stops generation


def make_responder(cfg: dict):
    l, p = cfg.get("llm", {}), cfg.get("persona", {})
    if l.get("backend", "transformers") == "template":
        return TemplateResponder(p.get("name", "Pip"))
    return TransformersResponder(l["model"], l.get("device", "auto"), l.get("quantization", "none"),
                                 l.get("max_new_tokens", 60), l.get("temperature", 0.7), l.get("top_p", 0.9),
                                 p.get("name", "Pip"), p.get("history_turns", 3))
