#!/usr/bin/env python3
"""Serve Unreal's rendered frames as an MJPEG stream over a single TCP port.

WebRTC/Pixel Streaming cannot traverse this host's SSH-only firewall, because
its media path is UDP on dynamic ports. MJPEG over HTTP needs exactly one TCP
port, which forwards cleanly:

    ssh -L 8090:localhost:8090 <user>@<host>

Set UE_SSH_TARGET to have the printed tunnel hint name your actual host.

The frame source is intentionally swappable. Today it tails the PNGs that
`-dumpmovie` writes; once the project has a SceneCapture2D feeding frames over
shared memory, only FrameSource needs to change.

Frames are pruned as they are served: -dumpmovie writes ~1MB per frame and this
volume is shared with 21 other users.
"""
import argparse
import io
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

BOUNDARY = "frameboundary"

PAGE = b"""<!doctype html>
<title>Agent view</title>
<style>
  body { margin:0; background:#111; color:#ddd; font-family:system-ui,sans-serif; }
  header { padding:8px 12px; font-size:13px; display:flex; gap:8px;
           align-items:center; flex-wrap:wrap; }
  button { background:#2a2a2e; color:#eee; border:1px solid #444;
           border-radius:5px; padding:5px 11px; font-size:13px; cursor:pointer; }
  button:hover { background:#37373c; }
  #status { color:#8a8; }
  .sep { color:#777; }
  img { display:block; width:100vw; height:auto; }
</style>
<header>
  <strong>Live view</strong>
  <button onclick="send('toggle')">Toggle camera</button>
  <button onclick="send('birdseye')">Bird's eye</button>
  <button onclick="send('overlay')">Overlay</button>
  <span class="sep">| free camera:</span>
  <button onclick="send('up')">&uarr; up</button>
  <button onclick="send('down')">&darr; down</button>
  <button onclick="send('fwd')">fwd</button>
  <button onclick="send('back')">back</button>
  <button onclick="send('left')">left</button>
  <button onclick="send('right')">right</button>
  <span id="status"></span>
</header>
<img src="/stream.mjpg" alt="live frame">
<script>
async function send(what) {
  const s = document.getElementById('status');
  s.textContent = what + '...';
  try {
    const r = await fetch('/control/' + what, {method: 'POST'});
    s.textContent = await r.text();
  } catch (e) { s.textContent = 'error: ' + e.message; }
}
</script>
"""


class FrameSource:
    """Newest complete frame from the dump directory, as JPEG bytes."""

    def __init__(self, directory: Path, pattern: str, keep: int, quality: int):
        self.directory = directory
        self.pattern = pattern
        self.keep = keep
        self.quality = quality
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._count = 0

    def _newest_complete(self):
        files = sorted(self.directory.glob(self.pattern))
        if len(files) < 2:
            return None
        # Skip the newest: Unreal may still be writing it.
        return files[-2]

    def _prune(self, keep_path: Path):
        files = sorted(self.directory.glob(self.pattern))
        for stale in files[: -self.keep] if len(files) > self.keep else []:
            if stale != keep_path:
                try:
                    stale.unlink()
                except OSError:
                    pass

    def poll_forever(self, interval: float):
        last = None
        while True:
            try:
                path = self._newest_complete()
                if path is not None and path != last:
                    with Image.open(path) as im:
                        buf = io.BytesIO()
                        im.convert("RGB").save(buf, "JPEG", quality=self.quality)
                    with self._lock:
                        self._jpeg = buf.getvalue()
                        self._count += 1
                    last = path
                    self._prune(path)
            except (OSError, ValueError):
                pass  # frame half-written; try again next tick
            time.sleep(interval)

    def latest(self):
        with self._lock:
            return self._jpeg, self._count


class Handler(BaseHTTPRequestHandler):
    source: FrameSource = None  # set on the class before serving
    bridge = None  # ControlBridge, or None when --no-control

    def log_message(self, *args):
        pass  # keep the console readable

    def do_POST(self):
        if not self.path.startswith("/control/"):
            self.send_error(404)
            return
        action = self.path.rsplit("/", 1)[-1]
        if self.bridge is None or not self.bridge.ready:
            self._text(503, "control bridge not ready")
            return
        try:
            if action == "toggle":
                on = self.bridge.toggle_camera()
                self._text(200, "debug camera ON" if on else "following agent")
            elif action == "birdseye":
                self.bridge.birdseye()
                self._text(200, "bird's eye")
            elif action == "overlay":
                self.bridge.hide_overlay()
                self._text(200, "toggled overlay")
            elif action in ("up", "down", "fwd", "back", "left", "right"):
                self._nudge(action)
                self._text(200, action)
            else:
                self._text(404, "unknown action")
        except Exception as exc:
            self._text(500, f"error: {exc}")

    def _nudge(self, action):
        b = self.bridge
        # The debug camera has no vertical movement bind available to us, so
        # altitude is changed by pitching and then moving along the view axis:
        # looking down, "back" rises and "forward" descends.
        if action == "up":
            b.pitch(10, down=True)
            b.hold("s", 0.5)
        elif action == "down":
            b.pitch(10, down=True)
            b.hold("w", 0.5)
        elif action == "fwd":
            b.hold("w", 0.5)
        elif action == "back":
            b.hold("s", 0.5)
        elif action == "left":
            b.hold("a", 0.5)
        elif action == "right":
            b.hold("d", 0.5)

    def _text(self, code, body):
        payload = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path.startswith("/stream.mjpg"):
            self._stream()
        elif self.path.startswith("/frame.jpg"):
            self._single()
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(PAGE)))
            self.end_headers()
            self.wfile.write(PAGE)

    def _single(self):
        jpeg, _ = self.source.latest()
        if jpeg is None:
            self.send_error(503, "no frame yet")
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg)))
        self.end_headers()
        self.wfile.write(jpeg)

    def _stream(self):
        self.send_response(200)
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header(
            "Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}"
        )
        self.end_headers()
        sent = -1
        try:
            while True:
                jpeg, count = self.source.latest()
                if jpeg is not None and count != sent:
                    sent = count
                    self.wfile.write(f"--{BOUNDARY}\r\n".encode())
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                else:
                    time.sleep(0.02)
        except (BrokenPipeError, ConnectionResetError):
            pass  # viewer closed the tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dir",
        default="/opt/Unrealprojects/render_test/Saved/Screenshots/LinuxEditor",
        help="directory Unreal writes frames into",
    )
    ap.add_argument("--pattern", default="MovieFrame*.png")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--keep", type=int, default=4, help="frames to leave on disk")
    ap.add_argument("--quality", type=int, default=80)
    ap.add_argument("--interval", type=float, default=0.05)
    ap.add_argument("--no-control", action="store_true",
                    help="serve frames only; no camera controls")
    args = ap.parse_args()

    directory = Path(args.dir)
    directory.mkdir(parents=True, exist_ok=True)

    source = FrameSource(directory, args.pattern, args.keep, args.quality)
    threading.Thread(
        target=source.poll_forever, args=(args.interval,), daemon=True
    ).start()

    Handler.source = source

    if not args.no_control:
        from control_bridge import ControlBridge

        bridge = ControlBridge()
        Handler.bridge = bridge

        def bring_up():
            try:
                bridge.start()
                print("control bridge ready (camera buttons live)")
            except Exception as exc:
                print(f"control bridge unavailable: {exc}")

        # Connecting costs ~15s of WebRTC negotiation; don't block frame serving.
        threading.Thread(target=bring_up, daemon=True).start()

    # Bind loopback only: reachable through the SSH tunnel, not from the network.
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"serving {directory}/{args.pattern} on http://127.0.0.1:{args.port}")
    print(f"tunnel:  ssh -L {args.port}:localhost:{args.port} "
          f"{os.environ.get('UE_SSH_TARGET', '<user>@<host>')}")
    server.serve_forever()


if __name__ == "__main__":
    main()
