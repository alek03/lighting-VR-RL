"""Runs INSIDE Unreal (start.sh loads it). Python outside Unreal controls the scene through it.

Unreal's Python only exists inside the editor, so unreal_bridge sends commands here over a
socket on this machine: one line of JSON in, one line of JSON out. Only the commands in
COMMANDS are accepted, never arbitrary code. The socket is checked once per rendered frame,
because Unreal's Python has to run on the editor's main thread. A command that needs frames to
render (capture) uses `yield`: it pauses there and continues on the next frame.
"""
import inspect
import json
import os
import socket
import sys
import traceback

import unreal

PROJECT_PYTHON = os.path.join(unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_dir()),
                              'Content', 'Python')
sys.path.insert(0, PROJECT_PYTHON)
import objectlab_bridge  # noqa: E402  the project's own code for putting objects on the table

TORCH = '/Game/BP_Torch.BP_Torch_C'
# Copied from the VR flashlight so the agent's light matches the human's. lighting_channels
# matters: table objects only receive light on channel 1, so without it the object is black.
TORCH_SETTINGS = ('intensity', 'intensity_units', 'attenuation_radius', 'inner_cone_angle',
                  'outer_cone_angle', 'source_radius', 'soft_source_radius', 'light_color',
                  'temperature', 'use_temperature', 'cast_shadows', 'use_inverse_squared_falloff',
                  'light_falloff_exponent', 'lighting_channels')

_camera = None   # SceneCapture2D actor
_target = None   # the render target (picture) it draws into
_light = None    # SpotLight actor


def _world():
    return unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()


def _spawn(cls):
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    return actors.spawn_actor_from_class(cls, unreal.Vector(0, 0, -10000))


def _project(command, **args):
    return objectlab_bridge.dispatch(dict(args, command=command, world='editor'))


def _export(path):
    directory, filename = os.path.split(path)
    unreal.RenderingLibrary.export_render_target(_world(), _target, directory, filename)


# -- commands ---------------------------------------------------------------------------------

def ping(req):
    return {}


def spawn(req):
    r = _project('spawn', object=req['object'], collection=req.get('collection', 'common'))
    return {'object': r['object_name'], 'centre': r['object_bounds_center_cm']}


def remove(req):
    _project('remove', collection=req.get('collection', 'common'))
    return {}


def set_camera(req):
    """Place the camera; the first call creates it. size = picture width and height in pixels."""
    global _camera, _target
    if not unreal.SystemLibrary.is_valid(_camera):
        _camera = _spawn(unreal.SceneCapture2D.static_class())
        comp = _camera.capture_component2d
        comp.set_editor_property('capture_source', unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
        comp.set_editor_property('capture_every_frame', True)
        comp.set_editor_property('always_persist_rendering_state', True)
    size = int(req['size'])
    if _target is None or _target.get_editor_property('size_x') != size:
        _target = unreal.RenderingLibrary.create_render_target2d(
            _world(), size, size, unreal.TextureRenderTargetFormat.RTF_RGBA8)
        _camera.capture_component2d.set_editor_property('texture_target', _target)
    pitch, yaw, roll = req['rotation']
    _camera.set_actor_location_and_rotation(unreal.Vector(*req['location']),
                                            unreal.Rotator(roll=roll, pitch=pitch, yaw=yaw), False, False)
    _camera.capture_component2d.set_editor_property('fov_angle', float(req['fov']))
    return {}


def set_light(req):
    """Put the flashlight at `location`, pointing at `target`; the first call creates it."""
    global _light
    if not unreal.SystemLibrary.is_valid(_light):
        _light = _spawn(unreal.SpotLight.static_class())
        comp = _light.spot_light_component
        comp.set_mobility(unreal.ComponentMobility.MOVABLE)
        # A Blueprint's settings can only be read from a placed copy: place one, read, delete.
        torch = _spawn(unreal.load_class(None, TORCH))
        try:
            src = torch.get_components_by_class(unreal.SpotLightComponent)[0]
            for name in TORCH_SETTINGS:
                comp.set_editor_property(name, src.get_editor_property(name))
        finally:
            torch.destroy_actor()
    loc, target = unreal.Vector(*req['location']), unreal.Vector(*req['target'])
    rot = unreal.MathLibrary.find_look_at_rotation(loc, target)
    _light.set_actor_location_and_rotation(loc, rot, False, False)
    return {}


def capture(req):
    """Save a picture to `path`, after `settle_frames` frames so the lighting can settle."""
    comp = _camera.capture_component2d
    for _ in range(int(req.get('settle_frames', 0))):
        comp.capture_scene()
        yield
    comp.capture_scene()
    _export(req['path'])
    return {}


def capture_mask(req):
    """Save a picture of the object alone, as depth (distance from the camera): empty space is 0,
    the object is not, whatever its colour or material. The camera's settings are restored after."""
    collection = req.get('collection', 'common')
    if collection == 'new':
        import objectlab_downloaded
        _, spawner, _ = objectlab_downloaded.context('editor')
    elif collection == 'common':
        _, spawner, _ = objectlab_bridge.context('editor')
    else:
        raise ValueError('collection must be common or new')
    obj = spawner.get_editor_property('CurrentObject')
    comp = _camera.capture_component2d
    try:
        comp.show_only_actor_components(obj)
        comp.set_editor_property('primitive_render_mode', unreal.SceneCapturePrimitiveRenderMode.PRM_USE_SHOW_ONLY_LIST)
        comp.set_editor_property('capture_source', unreal.SceneCaptureSource.SCS_DEVICE_DEPTH)
        yield
        comp.capture_scene()
        _export(req['path'])
    finally:
        comp.set_editor_property('primitive_render_mode', unreal.SceneCapturePrimitiveRenderMode.PRM_RENDER_SCENE_PRIMITIVES)
        comp.set_editor_property('capture_source', unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
        comp.clear_show_only_components()
    return {}


def cleanup(req):
    """Remove everything we added: object, camera, light."""
    global _camera, _target, _light
    _project('remove', collection=req.get('collection', 'common'))
    for actor in (_camera, _light):
        if unreal.SystemLibrary.is_valid(actor):
            actor.destroy_actor()
    if _target is not None:
        unreal.RenderingLibrary.release_render_target2d(_target)
    _camera = _target = _light = None
    return {}


COMMANDS = {f.__name__: f for f in (ping, spawn, remove, set_camera, set_light, capture, capture_mask, cleanup)}


# -- the socket, checked once per frame ---------------------------------------------------------

def _run(line):
    """Start one command. Returns the reply, or a generator if it needs more frames."""
    try:
        req = json.loads(line)
        if req.get('command') not in COMMANDS:
            raise ValueError('unknown command %r; allowed: %s' % (req.get('command'), ', '.join(COMMANDS)))
        return COMMANDS[req['command']](req)
    except Exception as exc:
        return _error(exc)


def _error(exc):
    unreal.log_warning('server: ' + traceback.format_exc())
    return {'ok': False, 'error': '%s: %s' % (type(exc).__name__, exc)}


class Server:
    def __init__(self, port):
        self.listener = socket.socket()
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(('127.0.0.1', port))
        self.listener.listen()
        self.listener.setblocking(False)
        self.clients = {}   # connection -> [unread bytes, command in progress or None]

    def reply(self, conn, result):
        result.setdefault('ok', True)
        conn.setblocking(True)
        conn.sendall(json.dumps(result).encode() + b'\n')
        conn.setblocking(False)

    def tick(self, _seconds):
        try:
            conn, _ = self.listener.accept()
            conn.setblocking(False)
            self.clients[conn] = [b'', None]
        except BlockingIOError:
            pass
        for conn, state in list(self.clients.items()):
            try:
                if state[1] is not None:                      # a command waiting for frames
                    try:
                        next(state[1])
                        continue
                    except StopIteration as done:
                        result = done.value
                    except Exception as exc:
                        result = _error(exc)
                    state[1] = None
                    self.reply(conn, result)
                try:
                    data = conn.recv(65536)
                    if not data:
                        raise ConnectionResetError
                    state[0] += data
                except BlockingIOError:
                    pass
                while state[1] is None and b'\n' in state[0]:
                    line, state[0] = state[0].split(b'\n', 1)
                    result = _run(line)
                    if inspect.isgenerator(result):
                        state[1] = result
                    else:
                        self.reply(conn, result)
            except OSError:
                conn.close()
                del self.clients[conn]


# Running this file again (e.g. after editing it) replaces the old server instead of adding one.
if hasattr(unreal, '_lighting_rl_server'):
    old, handle = unreal._lighting_rl_server
    unreal.unregister_slate_post_tick_callback(handle)
    old.listener.close()
_server = Server(int(os.environ.get('LIGHTING_RL_PORT', 6780)))
unreal._lighting_rl_server = (_server, unreal.register_slate_post_tick_callback(_server.tick))
unreal.log('server: listening on 127.0.0.1:%d' % _server.listener.getsockname()[1])
