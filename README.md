# CCTV-mcp
 hybrid CCTV pipeline (YOLO fast + OpenAI Vision slow) wired as an MCP server.
Here’s a minimal, opinionated repo you can drop into GitHub for a hybrid CCTV pipeline (YOLO fast + OpenAI Vision slow) wired as an MCP server.

---

📁 Repository layout

hybrid-cctv-agent/
├─ README.md
├─ requirements.txt
├─ mcp_server/
│  ├─ __init__.py
│  ├─ server.py
│  ├─ cctv_session.py
│  ├─ vision_fast_yolo.py
│  ├─ vision_slow_openai.py
│  └─ event_store.py
└─ agent/
   ├─ prompt.md
   └─ tools.json


---

📦 `requirements.txt`

opencv-python
ultralytics          # YOLOv8/9
numpy
openai
fastapi
uvicorn
pydantic


---

🧠 `mcp_server/vision_fast_yolo.py` (fast detector)

from ultralytics import YOLO
import numpy as np

class FastDetector:
    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.4):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame: np.ndarray):
        results = self.model.predict(frame, conf=self.conf, verbose=False)[0]
        events = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            events.append({
                "label": label,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
            })
        return events


---

🧠 `mcp_server/vision_slow_openai.py` (OpenAI Vision)

from openai import OpenAI

class SlowVision:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model

    def analyze_frame(self, jpeg_bytes: bytes, prompt: str):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "input_image", "image": jpeg_bytes},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return resp.choices[0].message.content


---

🎥 `mcp_server/cctv_session.py` (sampling + hybrid logic)

import cv2
import threading
import time
from .vision_fast_yolo import FastDetector
from .vision_slow_openai import SlowVision
from .event_store import EventStore

class CctvSession:
    def __init__(self, stream_url: str, session_id: str):
        self.stream_url = stream_url
        self.session_id = session_id
        self.cap = cv2.VideoCapture(stream_url)
        self.fast = FastDetector()
        self.slow = SlowVision()
        self.events = EventStore()
        self.active = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.active:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.5)
                continue

            ts = time.time()
            fast_events = self.fast.detect(frame)

            # store fast events
            for ev in fast_events:
                self.events.add_event(self.session_id, {
                    "time": ts,
                    "type": "fast_detection",
                    **ev,
                })

            # trigger slow analysis only on “interesting” events
            if any(e["label"] in ("person", "car", "truck") for e in fast_events):
                _, buf = cv2.imencode(".jpg", frame)
                jpeg = buf.tobytes()
                desc = self.slow.analyze_frame(
                    jpeg,
                    "Describe what is happening in this frame in one sentence."
                )
                self.events.add_event(self.session_id, {
                    "time": ts,
                    "type": "slow_analysis",
                    "description": desc,
                })

            time.sleep(0.3)  # sampling interval

    def stop(self):
        self.active = False
        self.cap.release()


---

🧾 `mcp_server/event_store.py` (rolling buffer)

import time
from collections import defaultdict, deque

class EventStore:
    def __init__(self, window_s: int = 600):
        self.window_s = window_s
        self.events = defaultdict(lambda: deque())

    def add_event(self, session_id: str, event: dict):
        self.events[session_id].append(event)
        self._prune(session_id)

    def get_recent(self, session_id: str, window_s: int):
        cutoff = time.time() - window_s
        return [e for e in self.events[session_id] if e["time"] >= cutoff]

    def _prune(self, session_id: str):
        cutoff = time.time() - self.window_s
        dq = self.events[session_id]
        while dq and dq[0]["time"] < cutoff:
            dq.popleft()


---

🌐 `mcp_server/server.py` (MCP‑style HTTP tools)

This is a simple FastAPI façade you can adapt to your MCP wiring.

from fastapi import FastAPI
from pydantic import BaseModel
from .cctv_session import CctvSession
from .event_store import EventStore

app = FastAPI()
sessions = {}
store = EventStore()

class StartWatchRequest(BaseModel):
    stream_url: str
    session_id: str

class GetEventsRequest(BaseModel):
    session_id: str
    window_s: int = 300

@app.post("/start_stream_watch")
def start_stream_watch(req: StartWatchRequest):
    if req.session_id in sessions:
        return {"status": "already_running"}
    sessions[req.session_id] = CctvSession(req.stream_url, req.session_id)
    return {"status": "ok", "session_id": req.session_id}

@app.post("/get_recent_events")
def get_recent_events(req: GetEventsRequest):
    return {
        "session_id": req.session_id,
        "events": store.get_recent(req.session_id, req.window_s),
    }


Run:

uvicorn mcp_server.server:app --reload


---

🤖 `agent/prompt.md` (Agent behavior)

You are a Live CCTV Hybrid Agent.

You have tools:
- start_stream_watch(stream_url, session_id)
- get_recent_events(session_id, window_s)

Use them to:
- start watching a camera when the user gives a URL
- summarize what happened recently
- answer “what’s happening now” from the latest events
- mention timestamps and whether info is from fast or slow analysis


---

📘 `README.md` (short)

# Hybrid CCTV Agent (YOLO + OpenAI Vision)

This repo implements a hybrid pipeline:

- **Fast path:** YOLO (local) for real-time people/vehicle/object detection.
- **Slow path:** OpenAI Vision (`gpt-4o-mini`) for semantic understanding.

The MCP-style server:
- opens a CCTV stream
- samples frames
- runs fast detection on every frame
- triggers slow OpenAI Vision analysis only on interesting events
- stores events in a rolling buffer for an OpenAI Agent to query

## Quick start

1. Install deps:

   ```bash
   pip install -r requirements.txt


1. Set OPENAI_API_KEY in your environment.
2. Run the server:uvicorn mcp_server.server:app --reload

3. Wire the HTTP endpoints as tools in your OpenAI Agents SDK config.



---

If you tell me your exact stack (pure MCP JSON over stdio vs HTTP, Node vs Python), I can tighten this into a drop‑in repo for that environment.