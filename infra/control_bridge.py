#!/usr/bin/env python3
"""Persistent input channel into the running Unreal instance.

Both console-command routes are closed on this build (RemoteControl hard-blocks
ExecuteConsoleCommand; Pixel Streaming's emitConsoleCommand is not wired
through), so the only way in is synthetic keyboard/mouse events over Pixel
Streaming's data channel.

A headless Chromium is held open for the session rather than relaunched per
command, because connecting and negotiating WebRTC costs ~15s.
"""
import json
import subprocess
import threading
import time

import requests
import websocket

PAGE_URL = (
    "http://127.0.0.1:8080/"
    "?AutoConnect=true&AutoPlayVideo=true&StartVideoMuted=true"
)

KEYS = {
    "w": ("KeyW", "w", 87),
    "a": ("KeyA", "a", 65),
    "s": ("KeyS", "s", 83),
    "d": ("KeyD", "d", 68),
    "semicolon": ("Semicolon", ";", 186),
    "backspace": ("Backspace", "Backspace", 8),
}


class ControlBridge:
    def __init__(self, debug_port=9230):
        self.debug_port = debug_port
        self.proc = None
        self.ws = None
        self._lock = threading.Lock()
        self._msg_id = 0
        self.ready = False
        self.debug_camera = False

    # -- plumbing -----------------------------------------------------------
    def _cdp(self, method, params=None):
        self._msg_id += 1
        mid = self._msg_id
        self.ws.send(json.dumps({"id": mid, "method": method,
                                 "params": params or {}}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == mid:
                return msg.get("result", {})

    def start(self):
        self.proc = subprocess.Popen(
            [
                "chromium", "--headless", "--no-sandbox", "--disable-dev-shm-usage",
                f"--remote-debugging-port={self.debug_port}",
                "--remote-allow-origins=*", "--window-size=1280,720",
                "--autoplay-policy=no-user-gesture-required", "about:blank",
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for _ in range(40):
            try:
                requests.get(f"http://127.0.0.1:{self.debug_port}/json/version",
                             timeout=1)
                break
            except requests.RequestException:
                time.sleep(0.5)
        else:
            raise RuntimeError("chromium did not start")

        targets = requests.get(f"http://127.0.0.1:{self.debug_port}/json").json()
        page = next(t for t in targets if t["type"] == "page")
        self.ws = websocket.create_connection(page["webSocketDebuggerUrl"],
                                              timeout=60)
        self._cdp("Page.enable")
        self._cdp("Runtime.enable")
        self._cdp("Page.navigate", {"url": PAGE_URL})

        probe = ("(() => { const v=document.querySelector('video');"
                 " return v?v.videoWidth:0; })()")
        deadline = time.time() + 60
        while time.time() < deadline:
            r = self._cdp("Runtime.evaluate",
                          {"expression": probe, "returnByValue": True})
            if r.get("result", {}).get("value"):
                break
            time.sleep(2)
        else:
            raise RuntimeError("pixel streaming video never started")

        # Input is only forwarded once the video element has focus.
        for t in ("mousePressed", "mouseReleased"):
            self._cdp("Input.dispatchMouseEvent",
                      {"type": t, "x": 640, "y": 360,
                       "button": "left", "clickCount": 1})
        self.ready = True

    def stop(self):
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass
        if self.proc:
            self.proc.kill()

    # -- input --------------------------------------------------------------
    def key(self, name, down):
        code, key, vk = KEYS[name]
        self._cdp("Input.dispatchKeyEvent", {
            "type": "keyDown" if down else "keyUp",
            "code": code, "key": key,
            "windowsVirtualKeyCode": vk, "nativeVirtualKeyCode": vk,
        })

    def tap(self, name):
        with self._lock:
            self.key(name, True)
            time.sleep(0.05)
            self.key(name, False)

    def hold(self, name, seconds):
        with self._lock:
            self.key(name, True)
            time.sleep(seconds)
            self.key(name, False)

    def mouse_to(self, x, y):
        self._cdp("Input.dispatchMouseEvent",
                  {"type": "mouseMoved", "x": int(x), "y": int(y)})

    # -- views --------------------------------------------------------------
    def toggle_camera(self):
        """Semicolon is bound to ToggleDebugCamera in the engine's BaseInput.ini."""
        self.tap("semicolon")
        self.debug_camera = not self.debug_camera
        return self.debug_camera

    def hide_overlay(self):
        self.tap("backspace")

    def _sweep(self, axis, steps, direction):
        """Drag the cursor along one axis to turn the view.

        Pixel Streaming forwards absolute cursor positions, not deltas, so a
        turn is a repeated sweep away from centre followed by a reset.
        """
        with self._lock:
            for i in range(1, steps + 1):
                offset = direction * 20 * i
                if axis == "y":
                    self.mouse_to(640, max(20, min(700, 360 + offset)))
                else:
                    self.mouse_to(max(20, min(1260, 640 + offset)), 360)
                time.sleep(0.04)
            self.mouse_to(640, 360)

    def pitch(self, steps, down=True):
        self._sweep("y", steps, 1 if down else -1)

    def yaw(self, steps, right=True):
        self._sweep("x", steps, 1 if right else -1)

    def birdseye(self):
        """Rise to a top-down view centred on wherever the agent currently is.

        There is no absolute-positioning command available to us, so this is
        relative motion and only repeatable if it starts from a known state.
        Toggling the debug camera off and on snaps it back to the player, which
        is that known state -- without this reset the result depends on wherever
        the camera happened to be left, which lands outside the arena.

        Centring on the agent rather than a fixed corner is also deliberate: the
        agent moves, and a corner view loses it.
        """
        if self.debug_camera:
            self.toggle_camera()          # back to the character
            time.sleep(0.8)
        self.toggle_camera()              # detach again, now at the player
        time.sleep(0.8)

        with self._lock:
            # Pitch down toward straight-down. Pixel Streaming sends absolute
            # cursor positions, so sweep downward repeatedly rather than once.
            for _ in range(3):
                for y in range(300, 700, 25):
                    self.mouse_to(640, y)
                    time.sleep(0.03)
        # Looking down, "back" translates to straight up.
        self.hold("s", 2.6)
        return True
