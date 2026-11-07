# unreal_bridge/\_\_init\_\_.py

The Python side of the Unreal bridge: the `Unreal` class that vision and RL code use to control
the scene. Unreal must be running first (`unreal_bridge/start.sh`). Run your code from the repo
folder with `.venv/bin/python -m ...` so `from unreal_bridge import Unreal` works.

```python
from unreal_bridge import Unreal

with Unreal() as ue:                          # connect; on exit, removes what it added
    ue.spawn("Mug")
    ue.set_light(azimuth=90, elevation=45)
    image = ue.capture()                      # numpy RGB array, 640 x 640
    box = ue.true_box()                       # [x0, y0, x1, y1] around the object
```

## The Unreal class

| Call | What it does |
|---|---|
| `Unreal()` | Connects to Unreal and places the camera from `config.json`. `Unreal(config=...)` uses a different settings dictionary. |
| `ue.objects()` | The object names that can be spawned (the project's `Config/objectlab_objects.json`). |
| `ue.spawn(name)` | Puts an object on the table, replacing any other, and returns its centre `[x, y, z]`. Waits until the object is actually drawn (the first use of an object takes a moment). Raises an error if it never appears (e.g. Thimble). If `camera.fill` is set, then aims and zooms at the object. |
| `ue.zoom_to_object(fill)` | Per-object zoom: turns the camera to the object and zooms until its larger side spans `fill` of the picture. The camera does not move. Called by `spawn` when `camera.fill` is set. |
| `ue.remove()` | Clears the table. |
| `ue.set_camera(location, rotation, fov)` | Moves, turns or zooms the camera. Anything left out comes from `config.json`. `ue.camera` holds the current values. |
| `ue.set_light(azimuth, elevation, radius=None)` | Puts the flashlight on the dome around the object, pointing at it. Radius defaults to `light.radius_cm`. Returns the light's position. |
| `ue.set_light_at(location, target)` | Puts the flashlight at any point, pointing at `target`. |
| `ue.capture()` | Takes a picture: numpy RGB array, `image.size` square. |
| `ue.capture_mask()` | True where the object is: the object rendered alone as depth, so light, shadow and colour don't affect it. |
| `ue.true_box(mask=None)` | `[x0, y0, x1, y1]` around the object, from the mask. |
| `ue.call(command, **args)` | Sends one raw command to `server.py` (used by the calls above). |
| `ue.close()` | Removes the object, camera and light, and disconnects. Done automatically at the end of a `with` block. |

## Plain functions

| Function | What it does |
|---|---|
| `light_position(centre, camera_location, azimuth, elevation, radius)` | The point on the dome. Angles are relative to the camera: azimuth 0 = the camera's side of the object, 90 = the camera's right, 180 = behind; elevation 0 = level with the object's centre, 90 = straight above. |
| `look_at(src, dst)` | `[pitch, yaw, roll]` that points from `src` to `dst`. |
| `mask_box(mask)` | The box around the True pixels of a mask. |
| `CONFIG` | `config.json`, loaded once at import. |

## Settings used (config.json)

`unreal.port`, `unreal.project`, `camera` (`location`, `rotation`, `fov`, `fill`), `image`
(`size`, `supersample`, `settle_frames`), `light.radius_cm`.

## Good to know

- Pictures are rendered at `size x supersample` and shrunk, which gives smooth edges.
- Pictures pass through a temporary folder in `/dev/shm` (memory, not disk) and are deleted.
- Positions are Unreal centimetres, Z up.
