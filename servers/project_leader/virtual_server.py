import sys
import os
import time
import threading
import argparse
import queue

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..')
sys.path.insert(0, project_root)

from flask import Flask, Response, jsonify, request
import numpy as np
import cv2

from duckiebot.camera_driver.godot_camera_driver import GodotCameraDriver, GodotCameraConfig
from duckiebot.wheel_driver.godot_wheels_driver import GodotWheelsDriver
from duckiebot.wheel_driver.wheels_driver_abs import WheelPWMConfiguration
from launcher.ports import find_available_port
from servers.common import shutdown_cleanup, suppress_http_logs

import tasks.project.project_leader.agent as agent

INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Convoy — Lead (Sim)</title>
<style>
  body { margin:0; background:#111; color:#eee; font-family:sans-serif; }
  header { padding:12px 16px; background:#1a1a1a; border-bottom:1px solid #333; }
  main { display:flex; justify-content:center; padding:16px; }
  img.stream { max-width:100%; height:auto; border:1px solid #333; background:#000; }
</style></head>
<body>
  <header>Convoy — Lead Bot (Simulation)</header>
  <main><img class="stream" src="/video" alt="camera"></main>
</body></html>
"""

app        = Flask(__name__)
camera     = None
wheels     = None
stop_event = threading.Event()

_frame_queue = queue.Queue(maxsize=2)
_debug_info_lock = threading.Lock()
_debug_info = {}


def visualize(frame_bgr):
    if frame_bgr is None:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "Waiting for camera...", (160, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
        return blank

    display = frame_bgr.copy()
    h, w = display.shape[:2]
    with _debug_info_lock:
        info = _debug_info.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    state = info.get('state', 'INIT')
    cv2.putText(display, f"State: {state}", (10, 25), font, 0.6, (0, 255, 0), 2)
    cv2.putText(display, f"Base: {info.get('base_speed', 0.0):.2f} "
                         f"Steer: {info.get('steering', 0.0):+.2f}", (10, 50), font, 0.5, (0, 255, 0), 1)
    cv2.putText(display, f"L: {info.get('left_speed', 0.0):+.2f}  "
                         f"R: {info.get('right_speed', 0.0):+.2f}", (10, 70), font, 0.5, (0, 255, 0), 1)
    cv2.putText(display, f"Route: {info.get('route_idx', 0)}", (10, 90), font, 0.5, (0, 255, 0), 1)

    rl = info.get('red_line')
    if rl:
        cv2.putText(display, f"RedLine: w={rl[0]} d={rl[1]}", (10, h - 60),
                    font, 0.5, (0, 0, 255), 1)
    tag_ids = info.get('apriltag_ids', [])
    if tag_ids:
        cv2.putText(display, f"Tags: {tag_ids}", (10, h - 40), font, 0.5, (255, 100, 255), 1)
    cv2.putText(display, f"{info.get('fps', 0.0):.1f} FPS", (w - 100, 25), font, 0.5, (200, 200, 200), 1)
    return display


def generate_frames():
    import time
    while True:
        try:
            frame = _frame_queue.get(timeout=0.5)
            display = visualize(frame)
            ret, jpeg = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        except queue.Empty:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Waiting for frames...", (160, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
            ret, jpeg = cv2.imencode('.jpg', blank, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.05)
        except Exception as e:
            print(f'[VideoStream] Error: {e}')
            time.sleep(0.05)


@app.route('/')
def index():
    return INDEX_HTML


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    with _debug_info_lock:
        return jsonify(dict(_debug_info))


@app.route('/command', methods=['POST'])
def command():
    return jsonify({'status': 'ok'})


@app.route('/shutdown')
def shutdown():
    shutdown_cleanup(wheels, camera, stop_event)
    return jsonify({'status': 'ok'})


def main():
    global camera, wheels, stop_event

    ap = argparse.ArgumentParser(description='Project Lead Server — Godot Simulation')
    ap.add_argument('--port',       type=int, default=5000)
    ap.add_argument('--frame-port', type=int, default=5001)
    ap.add_argument('--wheel-port', type=int, default=5002)
    ap.add_argument('--godot-host', type=str, default='localhost')
    args = ap.parse_args()

    suppress_http_logs()
    print('=' * 60)
    print('PROJECT LEAD SERVER — GODOT SIMULATION')
    print('=' * 60)

    print('\n[1/3] Initializing wheels (Godot)...')
    wheels = GodotWheelsDriver(
        WheelPWMConfiguration(pwm_min=0), WheelPWMConfiguration(pwm_min=0),
        godot_host=args.godot_host, godot_port=args.wheel_port,
    )
    # Same map as project, but drop the NPC leader ahead on the path.
    time.sleep(0.3)
    wheels.remove_objects('npcpath')

    print('\n[2/3] Initializing camera (Godot)...')
    camera = GodotCameraDriver(godot_config=GodotCameraConfig(host='0.0.0.0', port=args.frame_port))
    camera.start()

    # Sim has no LED hardware and no wheel encoders; the lead agent handles
    # leds=None (LED commands are no-ops) and encoders=None (turns fall back to
    # lane-reacquire + timeout).
    print('\n[3/3] Starting lead agent...')
    stop_event.clear()
    threading.Thread(
        target=agent.main,
        args=(camera, wheels, None, stop_event, _frame_queue, _debug_info_lock, _debug_info),
        kwargs={'encoders': None},
        daemon=True,
        name='LeadAgentThread',
    ).start()
    print('  agent.main() running')

    web_port = find_available_port(args.port)
    print(f'\nWeb Interface: http://localhost:{web_port}')
    print('=' * 60 + '\n')

    try:
        app.run(host='127.0.0.1', port=web_port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print('\nShutting down...')
    finally:
        shutdown_cleanup(wheels, camera, stop_event)


if __name__ == '__main__':
    sys.exit(main())