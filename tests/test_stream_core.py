import numpy as np

from meld_robot.labels import EMO2ID
from meld_robot.stream.prosody import extract_prosody
from meld_robot.stream.state_machine import EmotionSmoother, Turn, TurnDetector, VadEvent, count_flips


def run(td, pattern, win=32.0):
    ev = []
    for p in pattern:
        e = td.update(p, win)
        if e:
            ev.append(e)
    return ev


def test_turn_start_and_endpoint():
    td = TurnDetector(min_speech_ms=300, endpoint_silence_ms=500)
    ev = run(td, [0.9] * 15 + [0.1] * 20)           # 480 ms speech, 640 ms silence
    assert ev == [VadEvent.SPEECH_START, VadEvent.SPEECH_END]


def test_short_burst_is_ignored():
    td = TurnDetector(min_speech_ms=300)
    assert run(td, [0.9] * 4 + [0.1] * 40) == []     # 128 ms click


def test_pause_inside_turn_does_not_end_it():
    td = TurnDetector(min_speech_ms=300, endpoint_silence_ms=500)
    ev = run(td, [0.9] * 15 + [0.1] * 8 + [0.9] * 15 + [0.1] * 20)  # 256 ms pause < 500 ms
    assert ev == [VadEvent.SPEECH_START, VadEvent.SPEECH_END]


def test_smoother_hysteresis_reduces_flips():
    rng = np.random.default_rng(0)
    seq = []
    for _ in range(40):
        p = np.array([0.30, 0.02, 0.02, 0.02, 0.34, 0.1, 0.2]) + rng.normal(0, 0.05, 7)
        p = np.clip(p, 0.001, None); seq.append(p / p.sum())
    raw = [int(p.argmax()) for p in seq]
    sm = EmotionSmoother(alpha=0.5, margin=0.1)
    smooth = [EMO2ID[sm.update(p)[0]] for p in seq]
    assert count_flips(smooth) < count_flips(raw)


def test_smoother_switches_on_clear_evidence():
    sm = EmotionSmoother(alpha=0.6, margin=0.1)
    neutral = np.eye(7)[EMO2ID["neutral"]] * 0.8 + 0.2 / 7
    anger = np.eye(7)[EMO2ID["anger"]] * 0.9 + 0.1 / 7
    for _ in range(3): sm.update(neutral)
    labels = [sm.update(anger)[0] for _ in range(6)]
    assert labels[-1] == "anger"


def _voiced(f0, dur=1.5, amp=0.3, sr=16000):
    t = np.arange(int(dur * sr)) / sr
    return (amp * np.sin(2 * np.pi * f0 * t)).astype(np.float32)


def test_prosody_pitch_and_energy_buckets():
    loud_high = extract_prosody(_voiced(260, amp=0.4), "hello there my friend")
    quiet_low = extract_prosody(_voiced(100, amp=0.01), "hello there my friend")
    assert loud_high.pitch == "high" and quiet_low.pitch == "low"
    assert loud_high.energy == "high" and quiet_low.energy == "low"
    assert loud_high.pitch_var == "flat"            # pure tone has no pitch movement


def test_prosody_short_or_silent_audio_is_unknown():
    assert extract_prosody(np.zeros(1000, np.float32)).pitch == "unknown"
    assert extract_prosody(np.zeros(16000, np.float32)).energy == "unknown"
