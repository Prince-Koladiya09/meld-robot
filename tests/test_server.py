import json

import numpy as np
from fastapi.testclient import TestClient

from meld_robot.config import load_config
from meld_robot.respond.llm import TemplateResponder
from meld_robot.server.app import create_app
from meld_robot.stream.pipeline import Components
from meld_robot.stream.vad import EnergyVAD
from tests.test_session import FakeASR, FakeEngine, silence, speech


def test_websocket_stream_roundtrip(tmp_path):
    cfg = load_config("configs/stream.yaml")
    cfg["server"]["log_dir"] = str(tmp_path)
    comp = Components(EnergyVAD(-45), FakeASR(), FakeEngine(), TemplateResponder())
    with TestClient(create_app(cfg, comp)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert "<html" in client.get("/").text.lower()
        with client.websocket_connect("/stream") as ws:
            audio = np.concatenate([silence(0.3), speech(1.4), silence(1.0)])
            pcm = (audio * 32767).astype(np.int16)
            for i in range(0, len(pcm), 320):
                ws.send_bytes(pcm[i:i + 320].tobytes())
            types, final, reply = [], None, ""
            while True:
                e = ws.receive_json()
                types.append(e["type"])
                if e["type"] == "state_final":
                    final = e
                if e["type"] == "reply_token":
                    reply += e["token"]
                if e["type"] == "reply_done":
                    break
    assert final["emotion"]["label"] == "anger" and final["mismatch"]
    assert "state_update" in types and reply.strip() == e["text"]
    logs = list(tmp_path.glob("session_*.jsonl"))
    assert logs and any(json.loads(l)["type"] == "state_final" for l in logs[0].read_text().splitlines())
