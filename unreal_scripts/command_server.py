"""Localhost command server that runs inside the Unreal editor.

Load it when launching the editor (infra/start_editor.sh does this):

    UnrealEditor <project>.uproject <map> ... -ExecCmds="py <repo>/unreal_scripts/command_server.py"

(-ExecutePythonScript would run this and then quit the editor.)

Why not Epic's Python remote execution, which the tufaelz project already enables:
its discovery is UDP multicast, and with RemoteExecutionMulticastBindAddress=127.0.0.1
the editor never receives discovery packets on Linux (a socket bound to a unicast
address does not get multicast traffic there; Windows delivers it anyway). Binding
to 0.0.0.0 would fix discovery but expose arbitrary Python execution to the network.

So this server listens on 127.0.0.1 only, speaks newline-delimited JSON, and accepts
only the named commands in COMMANDS -- never arbitrary code. It reuses the project's
own objectlab_bridge / objectlab_capture modules unchanged.

The socket is polled from the editor's tick, since Unreal's Python API must be called
on the game thread. A handler may be a generator: each `yield` lets one frame render
before it resumes, and its return value is sent as the reply. That is how `capture`
waits for lighting to settle without blocking the editor.
"""
import inspect
import json
import os
import socket
import sys
import time
import traceback

import unreal

PORT = int(os.environ.get('UE_COMMAND_PORT', '6780'))
PROJECT_PYTHON = os.path.join(unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_dir()),
                              'Content', 'Python')
if PROJECT_PYTHON not in sys.path:
    sys.path.insert(0, PROJECT_PYTHON)

import objectlab_bridge  # noqa: E402  (project module, needs the path above)
import objectlab_capture  # noqa: E402

MAX_REQUEST_BYTES = 1 << 20
TORCH_CLASS = '/Game/BP_Torch.BP_Torch_C'
OBJECT_MESHES = '/Game/CommonObjectsRealistic/Meshes/SM_'
# Spotlight settings copied from BP_Torch so the agent's light matches the human's torch.
# lighting_channels matters since the Oct 5 project update: table objects receive light on
# channel 1 only, and a new spotlight is on channel 0 only, so without it the object is black.
TORCH_LIGHT_PROPS = ('intensity', 'intensity_units', 'attenuation_radius', 'inner_cone_angle',
                     'outer_cone_angle', 'source_radius', 'soft_source_radius', 'light_color',
                     'temperature', 'use_temperature', 'cast_shadows',
                     'use_inverse_squared_falloff', 'light_falloff_exponent', 'lighting_channels')
REPORT_PROPS = ('mobility', 'intensity', 'intensity_units', 'attenuation_radius',
                'inner_cone_angle', 'outer_cone_angle', 'cast_shadows', 'temperature')

_light = None
_frame = 0
_frame_times = []


# -- helpers ---------------------------------------------------------------------------------

def _vec(v):
    return [round(v.x, 1), round(v.y, 1), round(v.z, 1)]


def _props(obj, names, as_text=True):
    out = {}
    for name in names:
        try:
            value = obj.get_editor_property(name)
            out[name] = str(value) if as_text else value
        except Exception:
            pass
    return out


def _world(req):
    world, _, name = objectlab_bridge.context(req.get('world', 'editor'))
    return world, name


def _spawn_actor(world, cls, transform):
    """Deferred spawn via reflection, so an explicit PIE/Simulate world is honoured."""
    gameplay = unreal.get_default_object(unreal.GameplayStatics)
    scale = unreal.SpawnActorScaleMethod.MULTIPLY_WITH_ROOT
    actor = gameplay.call_method('BeginDeferredActorSpawnFromClass', args=(
        world, cls, transform, unreal.SpawnActorCollisionHandlingMethod.ALWAYS_SPAWN, None, scale))
    if not unreal.SystemLibrary.is_valid(actor):
        raise RuntimeError('could not spawn %s' % cls.get_name())
    gameplay.call_method('FinishSpawningActor', args=(actor, transform, scale))
    return actor


def _read_torch(as_text):
    """Blueprint components are only readable on an instance: spawn one far away, then delete."""
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    torch = eas.spawn_actor_from_class(unreal.load_class(None, TORCH_CLASS), unreal.Vector(0, 0, -10000))
    try:
        comps = torch.get_components_by_class(unreal.SpotLightComponent)
        if not comps:
            raise RuntimeError('BP_Torch has no SpotLightComponent')
        return _props(comps[0], TORCH_LIGHT_PROPS, as_text=as_text)
    finally:
        eas.destroy_actor(torch)


# -- commands --------------------------------------------------------------------------------

def _ping(req):
    recent = [t for t in _frame_times if t > time.monotonic() - 2.0]
    fps = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 2 else None
    return {'ok': True, 'frame': _frame, 'fps': round(fps, 1) if fps else None}


def _status(req):
    return objectlab_bridge.dispatch(dict(req, command='status'))


def _spawn(req):
    return objectlab_bridge.dispatch(dict(req, command='spawn'))


def _remove(req):
    return objectlab_bridge.dispatch(dict(req, command='remove'))


def _capture_setup(req):
    objectlab_capture.setup(req.get('world', 'editor'), req['camera'],
                            int(req.get('width', 1280)), int(req.get('height', 720)))
    return {'ok': True}


def _capture(req):
    """Render `settle_frames` frames (so Lumen and temporal AA converge), then save a PNG."""
    actor = objectlab_capture._actor
    if not unreal.SystemLibrary.is_valid(actor):
        raise RuntimeError('call capture_setup first')
    comp = actor.capture_component2d
    settle = int(req.get('settle_frames', 0))
    for _ in range(settle):
        comp.capture_scene()
        yield
    comp.capture_scene()
    directory, filename = os.path.split(os.path.abspath(req['path']))
    objectlab_capture.export(directory, filename)
    return {'ok': True, 'path': os.path.join(directory, filename), 'settle_frames': settle}


def _capture_mask(req):
    """Save the object alone, unlit (its base colour on black), from the capture camera: its
    exact silhouette, independent of lighting and shadows. The camera's normal settings are
    restored afterwards."""
    actor = objectlab_capture._actor
    if not unreal.SystemLibrary.is_valid(actor):
        raise RuntimeError('call capture_setup first')
    _, spawner, _ = objectlab_bridge.context(req.get('world', 'editor'))
    obj = spawner.get_editor_property('CurrentObject')
    if not unreal.SystemLibrary.is_valid(obj):
        raise RuntimeError('no object on the table')
    comp = actor.capture_component2d
    saved = {k: comp.get_editor_property(k) for k in ('capture_source', 'primitive_render_mode')}
    try:
        comp.show_only_actor_components(obj)    # the show-only list cannot be set as a property
        comp.set_editor_property('primitive_render_mode', unreal.SceneCapturePrimitiveRenderMode.PRM_USE_SHOW_ONLY_LIST)
        comp.set_editor_property('capture_source', unreal.SceneCaptureSource.SCS_BASE_COLOR)
        yield                                   # one rendered frame in mask mode
        comp.capture_scene()
        directory, filename = os.path.split(os.path.abspath(req['path']))
        objectlab_capture.export(directory, filename)
    finally:
        for k, v in saved.items():
            comp.set_editor_property(k, v)
        comp.clear_show_only_components()
    return {'ok': True, 'path': os.path.join(directory, filename)}


def _scale_object(req):
    """Uniformly scale the object on the table. A workaround for models whose drawn geometry is
    out of sync with their stored bounds (see training/object_scale.json)."""
    _, spawner, _ = objectlab_bridge.context(req.get('world', 'editor'))
    obj = spawner.get_editor_property('CurrentObject')
    if not unreal.SystemLibrary.is_valid(obj):
        raise RuntimeError('no object on the table')
    s = float(req['scale'])
    if not 0.001 <= s <= 1000:
        raise ValueError('scale must be between 0.001 and 1000')
    obj.set_actor_scale3d(unreal.Vector(s, s, s))
    return {'ok': True, 'scale': s, 'location': _vec(obj.get_actor_location())}


def _capture_cleanup(req):
    objectlab_capture.cleanup()
    return {'ok': True}


def _light_setup(req):
    """Spawn the agent's flashlight: a movable spotlight with BP_Torch's light settings."""
    global _light
    _light_cleanup(req)
    world, world_name = _world(req)
    _light = _spawn_actor(world, unreal.SpotLight.static_class(),
                          unreal.Transform(location=unreal.Vector(0, 0, -10000)))
    comp = _light.spot_light_component
    comp.set_mobility(unreal.ComponentMobility.MOVABLE)
    applied = {}
    for name, value in _read_torch(as_text=False).items():
        try:
            comp.set_editor_property(name, value)
            applied[name] = str(value)
        except Exception:
            pass
    for name, value in (req.get('overrides') or {}).items():
        comp.set_editor_property(name, value)
        applied[name] = str(value)
    return {'ok': True, 'world': world_name, 'settings': applied}


def _set_light(req):
    """Place the flashlight at `location` and aim it at `target` (world cm)."""
    if not unreal.SystemLibrary.is_valid(_light):
        raise RuntimeError('call light_setup first')
    loc = unreal.Vector(*map(float, req['location']))
    target = unreal.Vector(*map(float, req['target']))
    rot = unreal.MathLibrary.find_look_at_rotation(loc, target)
    _light.set_actor_location_and_rotation(loc, rot, False, False)
    if 'intensity' in req:
        _light.spot_light_component.set_intensity(float(req['intensity']))
    return {'ok': True, 'location': _vec(loc), 'rotation': [round(rot.pitch, 2), round(rot.yaw, 2)],
            'frame': _frame}


def _light_cleanup(req):
    global _light
    if unreal.SystemLibrary.is_valid(_light):
        _light.destroy_actor()
    _light = None
    return {'ok': True}


def _set_cvars(req):
    """Set rendering console variables, e.g. {"r.TextureStreaming": 0}. Only `r.` names with
    plain numeric values are accepted, so this cannot run arbitrary console commands."""
    world, _ = _world(req)
    applied = {}
    for name, value in req['cvars'].items():
        if not name.startswith('r.') or not name.replace('.', '').replace('_', '').isalnum():
            raise ValueError('only r.* rendering cvars are allowed: %r' % name)
        if not isinstance(value, (int, float)):
            raise ValueError('cvar values must be numbers: %r' % (value,))
        unreal.SystemLibrary.execute_console_command(world, '%s %s' % (name, value))
        applied[name] = value
    return {'ok': True, 'applied': applied}


def _play(req):
    """Start or stop Simulate-in-Editor: gameplay runs, but no player or headset is spawned."""
    les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    action = req.get('action')
    if action == 'simulate':
        les.editor_play_simulate()
    elif action == 'stop':
        les.editor_request_end_play()
    else:
        raise ValueError("action must be 'simulate' or 'stop'")
    return {'ok': True, 'action': action}


def _describe(req):
    """Where things are: table, lights, nearby geometry, torch settings, object sizes."""
    world, spawner, world_name = objectlab_bridge.context(req.get('world', 'editor'))
    centre = spawner.get_actor_location()
    radius = float(req.get('nearby_cm', 600))
    result = {'ok': True, 'world': world_name, 'table_centre_cm': _vec(centre),
              'lights': [], 'nearby': [], 'post_process': []}
    for actor in unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor):
        for comp in actor.get_components_by_class(unreal.LightComponent):
            result['lights'].append(dict(actor=actor.get_actor_label(),
                                         location=_vec(comp.get_world_location()),
                                         **_props(comp, REPORT_PROPS)))
        if isinstance(actor, unreal.PostProcessVolume):
            # An exposure override here would undo AutoExposure=False in project config.
            result['post_process'].append(dict(actor=actor.get_actor_label(), **_props(
                actor.get_editor_property('settings'),
                ('override_auto_exposure_method', 'auto_exposure_method'))))
        if isinstance(actor, unreal.StaticMeshActor):
            origin, extent = actor.get_actor_bounds(False)
            dist = (origin - centre).length()
            if dist < radius:
                result['nearby'].append(dict(actor=actor.get_actor_label(), dist_cm=round(dist),
                                             centre=_vec(origin), half_size=_vec(extent)))
    result['nearby'].sort(key=lambda d: d['dist_cm'])
    result['torch_light'] = _read_torch(as_text=True)
    result['objects'] = {}
    for name in objectlab_bridge.OBJECTS:
        bounds = unreal.load_asset(OBJECT_MESHES + name).get_bounds()
        e = bounds.box_extent
        result['objects'][name] = dict(size_cm=[round(2 * e.x, 1), round(2 * e.y, 1), round(2 * e.z, 1)],
                                       radius_cm=round(bounds.sphere_radius, 1))
    return result


COMMANDS = {
    'ping': _ping,
    'describe': _describe,
    'status': _status,
    'spawn': _spawn,
    'remove': _remove,
    'capture_setup': _capture_setup,
    'capture': _capture,
    'capture_mask': _capture_mask,
    'capture_cleanup': _capture_cleanup,
    'scale_object': _scale_object,
    'light_setup': _light_setup,
    'set_light': _set_light,
    'light_cleanup': _light_cleanup,
    'play': _play,
    'set_cvars': _set_cvars,
}


# -- server ----------------------------------------------------------------------------------

def _error(exc):
    unreal.log_warning('command_server: ' + traceback.format_exc())
    return {'ok': False, 'error': '%s: %s' % (type(exc).__name__, exc)}


class Server:
    def __init__(self, port):
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(('127.0.0.1', port))
        self.listener.listen(4)
        self.listener.setblocking(False)
        self.clients = {}  # socket -> {'buf': bytes, 'job': generator or None}

    def _start(self, line):
        """Run a request. Returns a reply dict, or a generator still in progress."""
        try:
            req = json.loads(line)
            handler = COMMANDS.get(req.get('command'))
            if handler is None:
                raise ValueError('unknown command %r; allowed: %s'
                                 % (req.get('command'), ', '.join(sorted(COMMANDS))))
            return handler(req)
        except Exception as exc:
            return _error(exc)

    def _send(self, conn, reply):
        conn.setblocking(True)
        try:
            conn.sendall(json.dumps(reply, default=str).encode() + b'\n')
        finally:
            conn.setblocking(False)

    def _drop(self, conn):
        conn.close()
        self.clients.pop(conn, None)

    def tick(self, _delta):
        global _frame
        _frame += 1
        _frame_times.append(time.monotonic())
        del _frame_times[:-120]
        try:
            while True:
                conn, _ = self.listener.accept()
                conn.setblocking(False)
                self.clients[conn] = {'buf': b'', 'job': None}
        except BlockingIOError:
            pass
        for conn, state in list(self.clients.items()):
            try:
                if state['job'] is not None:
                    try:
                        next(state['job'])
                        continue            # one step per frame
                    except StopIteration as done:
                        reply = done.value
                    except Exception as exc:
                        reply = _error(exc)
                    state['job'] = None
                    self._send(conn, reply)
                try:
                    data = conn.recv(65536)
                    if not data:
                        raise ConnectionResetError
                    state['buf'] += data
                except BlockingIOError:
                    pass
                if len(state['buf']) > MAX_REQUEST_BYTES:
                    raise ConnectionResetError
                # Requests on one connection run in order; a multi-frame job blocks later ones.
                while state['job'] is None and b'\n' in state['buf']:
                    line, state['buf'] = state['buf'].split(b'\n', 1)
                    result = self._start(line)
                    if inspect.isgenerator(result):
                        state['job'] = result
                    else:
                        self._send(conn, result)
            except OSError:
                self._drop(conn)


# Re-running the script (e.g. after edits) must not stack duplicate servers.
_previous = getattr(unreal, '_vrnav_command_server', None)
if _previous is not None:
    unreal.unregister_slate_post_tick_callback(_previous[1])
    _previous[0].listener.close()

_server = Server(PORT)
_handle = unreal.register_slate_post_tick_callback(_server.tick)
unreal._vrnav_command_server = (_server, _handle)
unreal.log('command_server: listening on 127.0.0.1:%d' % PORT)
