"""
Project Leader Server — REAL HARDWARE.

Lane-follow leader with red-line stops and ModCon odometry turns.
Dashboard: HSV sliders, start/stop, lane PID (same UI as simulation).

Launched on the bot after:
  python launch.py --run --bot <hostname> --task project_leader
"""

import argparse
import os
import signal
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

from duckiebot.camera_driver import CameraDriver
from duckiebot.led_driver import LEDDriver
from duckiebot.wheel_driver import DaguWheelsDriver
from duckiebot.wheel_driver.wheels_driver_abs import WheelPWMConfiguration
from launcher.ports import find_available_port
from servers.common import make_frame_generator, shutdown_cleanup, suppress_http_logs
from servers.templates.lane_servoing import LANE_SERVOING_TEMPLATE as HTML_TEMPLATE

import tasks.project_leader.packages.leader_agent as leader_agent
from tasks.project_leader.packages.leader_lane import LeaderLaneAgent

_CONFIG_FILE = 'leader_config.yaml'
leader_agent.CONFIG_FILE = _CONFIG_FILE
CONFIG_PATH = os.path.join(project_root, 'config', _CONFIG_FILE)
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
        # Hardware camera is BGR; Godot path uses RGB in the agent loop only.
        return frame if frame.ndim == 3 else frame
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(blank, 'Waiting for camera...', (160, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
    return blank


generate_frames = make_frame_generator(lambda: camera, _visualize, quality=70, rgb=False)


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
    if wheels is not None:
        st['encoders'] = getattr(wheels, 'encoders', None) is not None
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
    if leader_agent.LANE is not None:
        leader_agent.LANE.reset_steering_state()
    leader_agent.RUNNING = False
    if wheels:
        wheels.set_wheels_speed(0.0, 0.0)
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

    ap = argparse.ArgumentParser(description='Project Leader Server — Real Hardware')
    ap.add_argument('--port', type=int, default=5000)
    args = ap.parse_args()

    suppress_http_logs()
    print('=' * 60)
    print('PROJECT LEADER SERVER — REAL HARDWARE')
    print('=' * 60)

    if DaguWheelsDriver is None:
        print('ERROR: DaguWheelsDriver not available (not on Duckiebot hardware?)')
        return 1
    if CameraDriver is None:
        print('ERROR: CameraDriver not available (not on Duckiebot hardware?)')
        return 1

    _lane_ui = LeaderLaneAgent(config_path=LANE_CONFIG_PATH)

    print('\n[1/4] Initializing LED driver...')
    try:
        leds = LEDDriver()
        leds.all_off()
        print('  LEDs: ok')
    except Exception as e:
        print(f'  LEDs: not available ({e})')
        leds = None

    print('\n[2/4] Initializing wheels driver...')
    wheels = DaguWheelsDriver(WheelPWMConfiguration(), WheelPWMConfiguration())
    if getattr(wheels, 'encoders', None) is not None:
        print('  Wheels: ok (encoders active — ModCon odometry)')
    else:
        print('  Wheels: ok (WARNING: no encoders — turn/cross distance will be estimated)')

    print('\n[3/4] Initializing camera driver...')
    camera = CameraDriver()
    camera.start()
    print('  Camera: ok')

    print('\n[4/4] Starting leader agent...')
    leader_agent.CONFIG_FILE = _CONFIG_FILE
    stop_event.clear()
    threading.Thread(
        target=leader_agent.main,
        args=(camera, wheels, leds, stop_event),
        daemon=True,
        name='LeaderAgent',
    ).start()
    print('  leader_agent.main() running')

    def _shutdown(signum, frame):
        print('\nShutting down...')
        if leds:
            try:
                leds.all_off()
                leds.release()
            except Exception:
                pass
        shutdown_cleanup(wheels, camera, stop_event)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    web_port = find_available_port(args.port)
    print(f'\nWeb Interface: http://0.0.0.0:{web_port}')
    print('=' * 60 + '\n')

    try:
        app.run(host='0.0.0.0', port=web_port, debug=False, threaded=True)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if leds:
            try:
                leds.all_off()
                leds.release()
            except Exception:
                pass
        stop_event.set()
        shutdown_cleanup(wheels, camera, stop_event)

    return 0


if __name__ == '__main__':
    sys.exit(main())
