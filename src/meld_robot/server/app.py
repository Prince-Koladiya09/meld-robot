"""FastAPI app: WebSocket /stream (audio in, JSON events out), static UI at /, health at /health.

Protocol
  client -> server : binary frames (16 kHz, mono, int16 PCM, any size; 20 ms recommended)
                     text JSON: {"type":"reset"} clears dialogue memory
  server -> client : JSON events (see meld_robot.stream.pipeline)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from ..config import load_config
from ..stream.pipeline import Components, Session

WEB_DIR = Path(__file__).resolve().parents[3] / "web"


def create_app(cfg: dict, components: Components | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if components is None:
            from ..stream.factory import build_components
            print("Loading models (first run downloads weights)...")
            app.state.comp = build_components(cfg)
            print("Models ready.")
        else:
            app.state.comp = components
        yield

    app = FastAPI(title="MELD emotion-aware robot", lifespan=lifespan)
    log_dir = Path(cfg.get("server", {}).get("log_dir", "logs/sessions"))

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/")
    async def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.websocket("/stream")
    async def stream(ws: WebSocket):
        await ws.accept()
        out: asyncio.Queue = asyncio.Queue()
        session = Session(cfg, app.state.comp, out, realtime=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        log = open(log_dir / f"session_{int(time.time())}.jsonl", "w", encoding="utf-8")

        async def sender():
            while True:
                evt = await out.get()
                log.write(json.dumps(evt, ensure_ascii=False) + "\n")
                await ws.send_json(evt)

        send_task = asyncio.create_task(sender())
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes"):
                    await session.feed(msg["bytes"])
                elif msg.get("text"):
                    ctrl = json.loads(msg["text"])
                    if ctrl.get("type") == "reset":
                        session.reset()
        except WebSocketDisconnect:
            pass
        finally:
            send_task.cancel()
            await session.close()
            log.close()

    return app


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stream.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    s = cfg.get("server", {})
    uvicorn.run(create_app(cfg), host=s.get("host", "127.0.0.1"), port=s.get("port", 8000))


if __name__ == "__main__":
    main()
