"""
Project Leader Server — SIMULATION.

Same web dashboard as visual_lane_servoing (HSV sliders, start/stop, PID)
plus leader status. Launched by `python launch.py --sim --task project_leader`
"""

import argparse
import os
import socket
import sys
import threading
import time

import cv2
import numpy as np
import yaml
from flask import Flask, Response, jsonify, render_template_string, request

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..')
sys.path.insert(0, project_root)

from duckiebot.camera_driver.godot_camera_driver import GodotCameraDriver, GodotCameraConfig
from duckiebot.led_driver import VirtualLEDsDriver
from duckiebot.wheel_driver.godot_wheels_driver import GodotWheelsDriver
from duckiebot.wheel_driver.wheels_driver_abs import WheelPWMConfiguration
from launcher.config import GODOT_SCENES
from launcher.ports import find_available_port
from servers.common import make_frame_generator, shutdown_cleanup, suppress_http_logs
from servers.templates.lane_servoing import LANE_SERVOING_TEMPLATE as HTML_TEMPLATE

import tasks.project_leader.packages.leader_agent as leader_agent
from tasks.project_leader.packages.leader_lane import LeaderLaneAgent

_SIM_CONFIG_FILE = 'leader_config_sim.yaml'
leader_agent.CONFIG_FILE = _SIM_CONFIG_FILE
CONFIG_PATH = os.path.join(project_root, 'config', _SIM_CONFIG_FILE)
LANE_CONFIG_PATH = os.path.join(project_root, 'config', 'leader_lane_config.yaml')
LANE_HSV_CONFIG_FILE = os.path.join(project_root, 'config', 'lane_servoing_hsv_config.yaml')

app = Flask(__name__)
camera = None
wheels = None
leds = None
stop_event = threading.Event()
_lane_ui = None


def _get_student_module():
    from tasks.visual_lane_servoing.packages import visual_servoing_activity
    return visual_servoing_activity


def _load_lane_config():
    try:
        with open(LANE_CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _save_lane_config(data):
    try:
        with open(LANE_CONFIG_PATH, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    except Exception as e:
        print(f'[leader] Could not save lane config: {e}')


def _visualize(frame):
    debug = getattr(leader_agent, 'DEBUG_FRAME', None)
    if debug is not None:
        return debug
    if frame is not None:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.ndim == 3 else frame
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(blank, 'Waiting for camera...', (160, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
    return blank


generate_frames = make_frame_generator(lambda: camera, _visualize, quality=70, rgb=True)


@app.route('/')
def index():
    cfg = _lane_ui or leader_agent.LANE
    return render_template_string(
        HTML_TEMPLATE,
        config=cfg,
        hostname=socket.gethostname(),
    )


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    st = dict(getattr(leader_agent, 'STATUS', {}) or {})
    st['running'] = bool(getattr(leader_agent, 'RUNNING', False))
    return jsonify(st)


@app.route('/running')
def get_running():
    return jsonify({'running': bool(getattr(leader_agent, 'RUNNING', False))})


@app.route('/start', methods=['POST'])
def start():
    leader_agent.RUNNING = True
    print('[leader] Drive started')
    return jsonify({'status': 'running'})


@app.route('/stop', methods=['POST'])
def stop():
    leader_agent.RUNNING = False
    if wheels:
        wheels.set_wheels_speed(0.0, 0.0)
    print('[leader] Drive stopped')
    return jsonify({'status': 'stopped'})


@app.route('/reset', methods=['POST'])
def reset():
    if wheels is not None:
        try:
            wheels.reset_game()
        except Exception:
            wheels.change_scene(GODOT_SCENES['project_leader'])
    if leader_agent.LANE is not None:
        leader_agent.LANE.reset_steering_state()
    leader_agent.RUNNING = False
    return jsonify({'status': 'ok'})


@app.route('/update_config', methods=['POST'])
def update_config():
    data = request.json or {}
    lane = leader_agent.LANE
    if lane is None:
        return jsonify({'status': 'not_ready'}), 503

    lane.p_gain = float(data.get('k_d', lane.p_gain))
    lane.d_gain = float(data.get('k_phi', lane.d_gain))
    lane.base_speed = float(data.get('const', lane.base_speed))

    if leader_agent.CFG is not None:
        leader_agent.CFG.setdefault('lane', {})
        leader_agent.CFG['lane'].update({
            'p_gain': lane.p_gain,
            'd_gain': lane.d_gain,
            'base_speed': lane.base_speed,
        })

    saved = _load_lane_config()
    saved['p_gain'] = lane.p_gain
    saved['d_gain'] = lane.d_gain
    saved['base_speed'] = lane.base_speed
    _save_lane_config(saved)
    return jsonify({'status': 'ok'})


@app.route('/get_hsv')
def get_hsv():
    return jsonify(_get_student_module().get_hsv_bounds())


@app.route('/update_hsv', methods=['POST'])
def update_hsv():
    data = request.json or {}
    mod = _get_student_module()
    current = mod.get_hsv_bounds()
    current.update({k: int(v) for k, v in data.items()})
    mod.set_hsv_bounds(
        [current['yellow_lower_h'], current['yellow_lower_s'], current['yellow_lower_v']],
        [current['yellow_upper_h'], current['yellow_upper_s'], current['yellow_upper_v']],
        [current['white_lower_h'], current['white_lower_s'], current['white_lower_v']],
        [current['white_upper_h'], current['white_upper_s'], current['white_upper_v']],
    )
    try:
        with open(LANE_HSV_CONFIG_FILE, 'w') as f:
            yaml.dump(current, f, default_flow_style=False)
    except Exception as e:
        print(f'[leader] Could not save HSV config: {e}')
    return jsonify({'status': 'ok'})


@app.route('/shutdown')
def shutdown():
    stop_event.set()
    shutdown_cleanup(wheels, camera, stop_event)
    return jsonify({'status': 'ok'})


def main():
    global camera, wheels, leds, stop_event, _lane_ui

    ap = argparse.ArgumentParser(description='Project Leader Server — Simulation')
    ap.add_argument('--port', type=int, default=5000)
    ap.add_argument('--frame-port', type=int, default=5001)
    ap.add_argument('--wheel-port', type=int, default=5002)
    ap.add_argument('--godot-host', type=str, default='localhost')
    args = ap.parse_args()

    suppress_http_logs()
    print('=' * 60)
    print('PROJECT LEADER SERVER — SIMULATION')
    print('=' * 60)

    _lane_ui = LeaderLaneAgent(config_path=LANE_CONFIG_PATH)

    leds = VirtualLEDsDriver(debug=False)
    leds.all_off()

    wheels = GodotWheelsDriver(
        WheelPWMConfiguration(pwm_min=0), WheelPWMConfiguration(pwm_min=0),
        godot_host=args.godot_host, godot_port=args.wheel_port,
    )
    wheels.trim = 0

    camera = GodotCameraDriver(
        godot_config=GodotCameraConfig(host='0.0.0.0', port=args.frame_port),
    )
    camera.start()

    leader_agent.CONFIG_FILE = _SIM_CONFIG_FILE
    stop_event.clear()
    threading.Thread(
        target=leader_agent.main,
        args=(camera, wheels, leds, stop_event),
        daemon=True,
    ).start()

    web_port = find_available_port(args.port)
    print(f'\nWeb Interface: http://localhost:{web_port}')
    print('=' * 60)

    try:
        app.run(host='127.0.0.1', port=web_port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print('\nShutting down...')
    finally:
        stop_event.set()
        shutdown_cleanup(wheels, camera, stop_event)


if __name__ == '__main__':
    main()
