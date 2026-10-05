"""Streaming session: audio frames in -> JSON events out.

Event types (all dicts with a "type" key), pushed to `out` (an asyncio.Queue):
  turn          {"turn": idle|listening|thinking|responding, "turn_id"}
  state_update  partial transcript + smoothed emotion while the user is still speaking
  state_final   final transcript, fused emotion, sentiment, prosody tags, text/audio disagreement flag
  reply_token   one streamed piece of the robot's reply
  reply_done    full reply + latency breakdown
  barge_in / reply_cancelled   user interrupted the robot

Time-based decisions (hops, endpointing) use the *audio clock* (samples received), not the wall clock,
so replaying a file faster than real time behaves the same as live input.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any, Optional

import numpy as np

from ..config import get as cfg_get
from ..labels import EMOTION_TO_SENTIMENT, EMOTIONS
from .prosody import extract_prosody
from .state_machine import EmotionSmoother, Turn, TurnDetector, VadEvent
from .vad import WINDOW

SR = 16000
WIN_MS = WINDOW / SR * 1000.0  # 32 ms


class Components:
    """Bundle of the heavy objects shared by all sessions (loaded once)."""

    def __init__(self, vad: Any, asr: Any, engine: Any, responder: Any):
        self.vad, self.asr, self.engine, self.responder = vad, asr, engine, responder


class Session:
    def __init__(self, cfg: dict, comp: Components, out: "asyncio.Queue[dict]", realtime: bool = True):
        g = lambda k, d: cfg_get(cfg, k, d)
        self.comp, self.out, self.realtime = comp, out, realtime
        self.td = TurnDetector(g("vad.threshold", 0.5), g("turn.min_speech_ms", 300), g("turn.endpoint_silence_ms", 500))
        self.smoother = EmotionSmoother(g("streaming.ema_alpha", 0.5), g("streaming.switch_margin", 0.10))
        self.emo_hop, self.asr_hop = g("streaming.emotion_hop_ms", 250), g("streaming.asr_hop_ms", 750)
        self.window = int(g("streaming.window_s", 6.0) * SR)
        self.max_utt = int(g("turn.max_utterance_s", 20.0) * SR)
        self.ring_len = int((g("turn.pre_roll_ms", 300) + g("turn.min_speech_ms", 300) + 160) / 1000 * SR)
        self.persona = g("persona.name", "Pip")

        self.turn = Turn.IDLE
        self.turn_id = 0
        self.t_ms = 0.0
        self._rem = np.zeros(0, np.float32)           # samples not yet forming a full VAD window
        self._ring = deque(maxlen=self.ring_len // WINDOW + 1)
        self._utt: list[np.ndarray] = []
        self._utt_samples = 0
        self._last_emo = self._last_asr = 0.0
        self._partial = ""
        self._busy = False
        self._tasks: set[asyncio.Task] = set()
        self._reply_task: Optional[asyncio.Task] = None
        self.history: list[tuple[str, str]] = []

    # ------------------------------------------------------------------ helpers
    def _emit(self, **evt) -> None:
        self.out.put_nowait(evt)

    def _set_turn(self, t: Turn) -> None:
        self.turn = t
        self._emit(type="turn", turn=t.value, turn_id=self.turn_id, t_ms=round(self.t_ms))

    def _context(self) -> list[str]:
        return [f"{self.persona if w == 'robot' else 'User'}: {t}" for w, t in self.history[-2:]]

    def _spawn(self, coro) -> asyncio.Task:
        t = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)
        return t

    def _utterance(self) -> np.ndarray:
        return np.concatenate(self._utt) if self._utt else np.zeros(0, np.float32)

    # ------------------------------------------------------------------ input
    async def feed(self, pcm: bytes | np.ndarray) -> None:
        """Accept any amount of 16 kHz mono audio (int16 bytes or float32 array)."""
        if isinstance(pcm, (bytes, bytearray)):
            x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            x = pcm.astype(np.float32)
        self._rem = np.concatenate([self._rem, x])
        while len(self._rem) >= WINDOW:
            chunk, self._rem = self._rem[:WINDOW], self._rem[WINDOW:]
            await self._on_window(chunk)

    async def _on_window(self, chunk: np.ndarray) -> None:
        self.t_ms += WIN_MS
        ev = self.td.update(self.comp.vad.prob(chunk), WIN_MS)

        if self.turn == Turn.LISTENING:
            self._utt.append(chunk)
            self._utt_samples += len(chunk)
            if ev == VadEvent.SPEECH_END or self._utt_samples >= self.max_utt:
                self.td.reset()
                await self._end_turn()
                return
            await self._maybe_update()
            return

        self._ring.append(chunk)
        if ev == VadEvent.SPEECH_START:
            if self.turn in (Turn.THINKING, Turn.RESPONDING):
                self._cancel_reply()
            self.turn_id += 1
            self._utt = list(self._ring)
            self._utt_samples = sum(len(c) for c in self._utt)
            self.smoother.reset()
            self._partial = ""
            self._last_emo = self._last_asr = self.t_ms
            self._set_turn(Turn.LISTENING)

    def _cancel_reply(self) -> None:
        if self._reply_task and not self._reply_task.done():
            self._reply_task.cancel()
            self._emit(type="barge_in", turn_id=self.turn_id, t_ms=round(self.t_ms))

    # ------------------------------------------------------------------ partial updates
    async def _maybe_update(self) -> None:
        if self._busy or self.t_ms - self._last_emo < self.emo_hop:
            return
        run_asr = self.t_ms - self._last_asr >= self.asr_hop
        self._last_emo = self.t_ms
        if run_asr:
            self._last_asr = self.t_ms
        wav = self._utterance()[-self.window:]
        self._busy = True
        task = self._spawn(self._update(self.turn_id, wav, run_asr))
        if not self.realtime:
            await task

    async def _update(self, turn_id: int, wav: np.ndarray, run_asr: bool) -> None:
        try:
            lat: dict[str, float] = {}
            if run_asr:
                t0 = time.perf_counter()
                text = await asyncio.to_thread(self.comp.asr.transcribe, wav, False)
                lat["asr"] = round((time.perf_counter() - t0) * 1000, 1)
                if turn_id != self.turn_id or self.turn != Turn.LISTENING:
                    return
                self._partial = text
            t0 = time.perf_counter()
            res = await asyncio.to_thread(self.comp.engine.predict, self._partial, self._context(), wav)
            lat["encode_fuse"] = round((time.perf_counter() - t0) * 1000, 1)
            if turn_id != self.turn_id or self.turn != Turn.LISTENING:
                return
            label, conf, sm = self.smoother.update(res.probs)
            self._emit(type="state_update", turn_id=turn_id, t_ms=round(self.t_ms), partial_text=self._partial,
                       emotion={"label": label, "confidence": round(conf, 3),
                                "probs": {e: round(float(p), 3) for e, p in zip(EMOTIONS, sm)}},
                       modality_weight={"text": round(res.gate_text, 3), "audio": round(1 - res.gate_text, 3)},
                       latency_ms=lat)
        finally:
            self._busy = False

    # ------------------------------------------------------------------ end of turn
    async def _end_turn(self) -> None:
        wav, tid = self._utterance(), self.turn_id
        self._utt, self._utt_samples = [], 0
        self._set_turn(Turn.THINKING)
        self._reply_task = self._spawn(self._finalize_and_reply(tid, wav))
        if not self.realtime:
            await self._reply_task

    async def _finalize_and_reply(self, tid: int, wav: np.ndarray) -> None:
        t_end = time.perf_counter()
        try:
            while self._busy:                          # let an in-flight partial update finish first
                await asyncio.sleep(0.005)
            t0 = time.perf_counter()
            text = await asyncio.to_thread(self.comp.asr.transcribe, wav, True)
            lat = {"asr": round((time.perf_counter() - t0) * 1000, 1)}
            if not text.strip():
                self._emit(type="state_final", turn_id=tid, text="", no_speech=True, latency_ms=lat)
                self._set_turn(Turn.IDLE)
                return
            t0 = time.perf_counter()
            res = await asyncio.to_thread(self.comp.engine.predict, text, self._context(), wav)
            lat["encode_fuse"] = round((time.perf_counter() - t0) * 1000, 1)
            t0 = time.perf_counter()
            pros = extract_prosody(wav[-self.window:], text)
            lat["prosody"] = round((time.perf_counter() - t0) * 1000, 1)

            top = int(np.argmax(res.probs))
            label = EMOTIONS[top]
            text_label = EMOTIONS[int(np.argmax(res.text_probs))] if res.text_probs is not None else None
            state = {
                "type": "state_final", "turn_id": tid, "text": text,
                "emotion": {"label": label, "confidence": round(float(res.probs[top]), 3),
                            "probs": {e: round(float(p), 3) for e, p in zip(EMOTIONS, res.probs)}},
                "sentiment": EMOTION_TO_SENTIMENT[label],
                "prosody": pros.to_dict(), "prosody_summary": pros.summary(),
                "text_emotion": text_label,
                "mismatch": bool(text_label is not None and text_label != label),
                "modality_weight": {"text": round(res.gate_text, 3), "audio": round(1 - res.gate_text, 3)},
                "latency_ms": lat,
            }
            self._emit(**state)
            self.history.append(("user", text))
            await self._reply(tid, state, text, t_end)
        except asyncio.CancelledError:
            self._emit(type="reply_cancelled", turn_id=tid)
            raise

    async def _reply(self, tid: int, state: dict, text: str, t_end: float) -> None:
        pieces: list[str] = []
        t_first: Optional[float] = None
        agen = self.comp.responder.stream(state, text, self.history[:-1])
        try:
            async for tok in agen:
                if t_first is None:
                    t_first = time.perf_counter()
                    self._set_turn(Turn.RESPONDING)
                pieces.append(tok)
                self._emit(type="reply_token", turn_id=tid, token=tok)
        except asyncio.CancelledError:
            if pieces:
                self.history.append(("robot", "".join(pieces).strip() + " ..."))
            raise
        finally:
            await agen.aclose()
        full = "".join(pieces).strip()
        now = time.perf_counter()
        self.history.append(("robot", full))
        self._emit(type="reply_done", turn_id=tid, text=full,
                   latency_ms={"first_token": round(((t_first or now) - t_end) * 1000, 1),
                               "total": round((now - t_end) * 1000, 1)})
        self._set_turn(Turn.IDLE)

    # ------------------------------------------------------------------ lifecycle
    async def flush(self) -> None:
        """Wait for in-flight work (used by replay/demo/tests after the last audio frame)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def close(self) -> None:
        for t in list(self._tasks):
            t.cancel()
        await asyncio.gather(*list(self._tasks), return_exceptions=True)

    def reset(self) -> None:
        self.history.clear()
        self.smoother.reset()
