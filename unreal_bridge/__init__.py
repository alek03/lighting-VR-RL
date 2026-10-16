"""Control the Unreal scene from Python. Unreal must be running: unreal_bridge/start.sh.

    from unreal_bridge import Unreal

    with Unreal() as ue:
        ue.spawn("Mug")
        ue.set_light(azimuth=90, elevation=45)
        image = ue.capture()        # numpy RGB array
        box = ue.true_box()         # [x0, y0, x1, y1] around the object

Light angles are relative to the camera: azimuth 0 = on the camera's side of the object,
90 = the camera's right, 180 = behind the object; elevation 0 = level with the object's
centre, 90 = straight above it. Positions are Unreal centimetres (Z up).
"""
import json
import math
import os
import shutil
import socket
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

CONFIG = json.loads((Path(__file__).resolve().parents[1] / 'config.json').read_text())


def look_at(src, dst):
    """[pitch, yaw, roll] that points from src towards dst."""
    d = np.subtract(dst, src)
    return [math.degrees(math.atan2(d[2], math.hypot(d[0], d[1]))), math.degrees(math.atan2(d[1], d[0])), 0.0]


def light_position(centre, camera_location, azimuth, elevation, radius):
    """Point on the dome of `radius` around `centre`, angles relative to the camera (see top)."""
    centre = np.asarray(centre, float)
    to_camera = np.asarray(camera_location, float) - centre
    to_camera[2] = 0
    to_camera /= np.linalg.norm(to_camera)
    right = np.array([to_camera[1], -to_camera[0], 0.0])
    az, el = math.radians(azimuth), math.radians(elevation)
    direction = math.cos(el) * (math.cos(az) * to_camera + math.sin(az) * right) + [0, 0, math.sin(el)]
    return (centre + radius * direction).tolist()


def mask_box(mask):
    """[x0, y0, x1, y1] box around the True pixels of a mask (None if there are none)."""
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else None


class Unreal:
    def __init__(self, config=CONFIG):
        self.config = config
        self.sock = socket.create_connection(('127.0.0.1', config['unreal']['port']), timeout=120)
        self.buffer = b''
        self.tmp = tempfile.mkdtemp(prefix='lighting-rl-', dir='/dev/shm')
        self.centre = None     # centre of the object on the table
        self.camera = None
        self.set_camera()

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.close()

    def call(self, command, **args):
        """Send one command to server.py and return its reply."""
        self.sock.sendall(json.dumps(dict(args, command=command)).encode() + b'\n')
        while b'\n' not in self.buffer:
            data = self.sock.recv(65536)
            if not data:
                raise ConnectionError('Unreal closed the connection (did it crash? see logs/editor.log)')
            self.buffer += data
        line, self.buffer = self.buffer.split(b'\n', 1)
        reply = json.loads(line)
        if not reply['ok']:
            raise RuntimeError('Unreal: ' + reply['error'])
        return reply

    def objects(self):
        """Names of the objects that can be put on the table."""
        catalog = Path(self.config['unreal']['project']).parent / 'Config' / 'objectlab_objects.json'
        return json.loads(catalog.read_text())

    def spawn(self, name):
        """Put an object on the table (replacing any other); returns its centre. Waits until
        the object is drawn: the first time an object is used, Unreal takes a moment to prepare it."""
        self.centre = self.call('spawn', object=name)['centre']
        for _ in range(100):
            if self.capture_mask().any():
                return self.centre
            time.sleep(0.1)
        raise RuntimeError(name + ' did not appear in the camera view')

    def remove(self):
        self.call('remove')
        self.centre = None

    def set_camera(self, location=None, rotation=None, fov=None):
        """Move, turn or zoom the camera. Anything left out keeps its config.json value."""
        c = self.config['camera']
        self.camera = {'location': location or c['location'], 'rotation': rotation or c['rotation'],
                       'fov': fov or c['fov']}
        size = self.config['image']['size'] * self.config['image']['supersample']
        self.call('set_camera', size=size, **self.camera)

    def set_light(self, azimuth, elevation, radius=None):
        """Put the flashlight on the dome around the object, pointing at it."""
        radius = radius or self.config['light']['radius_cm']
        location = light_position(self.centre, self.camera['location'], azimuth, elevation, radius)
        self.set_light_at(location, self.centre)
        return location

    def set_light_at(self, location, target):
        """Put the flashlight at any point, pointing at `target`."""
        self.call('set_light', location=list(location), target=list(target))

    def capture(self):
        """Picture from the camera, as a numpy RGB array (size x size from config.json)."""
        path = self._render('capture', settle_frames=self.config['image']['settle_frames'])
        size = self.config['image']['size']
        # Rendered larger and shrunk: Unreal's scene captures get no edge smoothing otherwise.
        return np.asarray(Image.open(path).convert('RGB').resize((size, size), Image.LANCZOS))

    def capture_mask(self):
        """True where the object is (rendered alone as depth, so light, shadow and colour don't matter)."""
        path = self._render('capture_mask')
        s = self.config['image']['supersample']
        mask = np.asarray(Image.open(path).convert('RGB')).max(axis=2) > 0
        return mask.reshape(mask.shape[0] // s, s, mask.shape[1] // s, s).any(axis=(1, 3))

    def true_box(self, mask=None):
        """[x0, y0, x1, y1] pixel box around the object (None if it is not in view)."""
        return mask_box(self.capture_mask() if mask is None else mask)

    def _render(self, command, **args):
        path = os.path.join(self.tmp, command + '.png')
        if os.path.exists(path):
            os.remove(path)
        self.call(command, path=path, **args)
        for _ in range(600):                       # the file can land just after the reply
            if os.path.exists(path) and os.path.getsize(path) > 0:
                return path
            time.sleep(0.05)
        raise TimeoutError('Unreal did not write ' + path)

    def close(self):
        """Remove the object, camera and light we added, and disconnect."""
        try:
            self.call('cleanup')
        finally:
            self.sock.close()
            shutil.rmtree(self.tmp, ignore_errors=True)
