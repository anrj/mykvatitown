import sys
import os
import threading
import queue
import socket
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '', '..')
sys.path.insert(0, project_root)

from flask import Flask, Response, render_template_string, jsonify, request
import cv2
import numpy as np

from tasks.object_detection.packages.agent import ObjectDetectionAgent
from tasks.visual_lane_servoing.packages.agent import LaneServoingAgent
from tasks.convoying.packages.target_activity import TargetTracker
from tasks.convoying.packages.convoy_activity import ConvoyController
from tasks.convoying.packages.red_line_stop import RedLineGate

from servers.convoying.visualization import create_convoying_visualization
from servers.templates.convoying import CONVOYING_TEMPLATE as HTML_TEMPLATE

from duckiebot.wheel_driver.godot_wheels_driver import GodotWheelsDriver
from duckiebot.wheel_driver.wheels_driver_abs import WheelPWMConfiguration
from duckiebot.camera_driver.godot_camera_driver import GodotCameraDriver, GodotCameraConfig
from launcher.ports import find_available_port
from servers.common import make_frame_generator, shutdown_cleanup, suppress_http_logs


app = Flask(__name__)

camera = None
wheels = None
object_agent = None
lane_agent = None
tracker = None
convoy_controller = None
red_line_gate = None

running = False
stop_event = threading.Event()

# Async detection — keeps the video stream smooth while YOLO runs.
_frame_queue = queue.Queue(maxsize=1)
_detection_lock = threading.Lock()
_last_detections = []  # scaled to full camera resolution

_last_target = None
_last_valid_target = None
_last_command = None
_last_lane_left = 0.0
_last_lane_right = 0.0
_last_red_line_state = None


# ---------------------------------------------------------------------------
# Background detection thread
# ---------------------------------------------------------------------------

def _detection_loop():
    """Pull frames from the queue, run YOLO, store scaled detections."""
    global _last_detections

    while not stop_event.is_set():
        if object_agent is None or not object_agent.model_loaded:
            threading.Event().wait(0.1)
            continue

        try:
            frame_rgb, orig_h, orig_w = _frame_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        result = object_agent.detect(frame_rgb)
        if result is None:
            continue

        # Scale boxes from model input resolution back to full camera size.
        sx = orig_w / float(object_agent.img_size)
        sy = orig_h / float(object_agent.img_size)

        scaled = [
            ((int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)), score, class_id)
            for (x1, y1, x2, y2), score, class_id in result
        ]

        with _detection_lock:
            _last_detections = scaled


# ---------------------------------------------------------------------------
# Per-frame pipeline
# ---------------------------------------------------------------------------

def visualize(frame_rgb):
    """
    Called for every MJPEG frame. frame_rgb is RGB from the Godot camera.

    Pipeline:
        camera frame
        -> red line gate
        -> push downscaled copy to detection thread
        -> grab latest scaled detections
        -> blue truck fallback if YOLO misses truck
        -> target tracking
        -> lane servoing if allowed
        -> convoy controller
        -> wheel speeds
        -> browser visualization
    """
    global _last_target
    global _last_valid_target
    global _last_command
    global _last_lane_left
    global _last_lane_right
    global _last_red_line_state

    if frame_rgb is None:
        return _placeholder("Waiting for Godot camera...")

    image_height, image_width = frame_rgb.shape[:2]

    # Red line detection.
    # If close red line is detected, lane following is disabled for 5 seconds.
    if red_line_gate is not None:
        red_line_state = red_line_gate.update(frame_rgb)
    else:
        red_line_state = None

    _last_red_line_state = red_line_state
    lane_disabled = bool(red_line_state and red_line_state.lane_disabled)

    # Feed detection thread — drop frame if it is busy.
    if object_agent is not None and object_agent.model_loaded:
        small = cv2.resize(frame_rgb, (object_agent.img_size, object_agent.img_size))
        try:
            _frame_queue.put_nowait((small, image_height, image_width))
        except queue.Full:
            pass

    # Grab the most recent scaled YOLO detections.
    with _detection_lock:
        detections = list(_last_detections)

    # Simulation fallback: detect visible blue lead truck if YOLO misses it.
    # Use last VALID target, not last frame target, because last frame may be LOST.
    detections = _add_blue_truck_fallback(frame_rgb, detections, _last_valid_target)

    # Target tracking.
    if tracker is not None:
        target = tracker.update(
            detections=detections,
            image_height=image_height,
            image_width=image_width,
        )
    else:
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    if target.found:
        _last_valid_target = target

    # Lane servoing.
    # Important: when lane_disabled is True, we DO NOT use lane commands.
    # The controller will steer only toward the leader truck.
    if lane_agent is not None and not lane_disabled:
        try:
            lane_left, lane_right = lane_agent.compute_commands(frame_rgb)
        except Exception as e:
            print(f"[Convoying] Lane error: {e}")
            lane_left, lane_right = 0.0, 0.0
    else:
        lane_left, lane_right = 0.0, 0.0

    # Convoy controller.
    # Normal mode: lane_left/lane_right steer the bot.
    # Red-line mode: lane_disabled=True, controller ignores lane and follows leader only.
    command = convoy_controller.decide(
        target=target,
        lane_left=lane_left,
        lane_right=lane_right,
        image_width=image_width,
        lane_disabled=lane_disabled,
    )

    # Drive wheels.
    game_over = wheels.is_game_over() if wheels is not None else False

    if running and not game_over:
        wheels.set_wheels_speed(command.left_speed, command.right_speed)
    else:
        if wheels is not None:
            wheels.set_wheels_speed(0.0, 0.0)

    _last_target = target
    _last_command = command
    _last_lane_left = lane_left
    _last_lane_right = lane_right

    image_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    return create_convoying_visualization(
        image_bgr=image_bgr,
        lane_debug_info=lane_agent.last_debug_info if lane_agent else {},
        target=target,
        command=command,
        lane_left=lane_left,
        lane_right=lane_right,
        detections=detections,
    )


generate_frames = make_frame_generator(lambda: camera, visualize, quality=50)


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, hostname=socket.gethostname())


@app.route('/video')
def video():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )


@app.route('/start', methods=['POST'])
def start():
    global running

    running = True
    print("[Convoying] Started")
    return jsonify({'status': 'running'})


@app.route('/stop', methods=['POST'])
def stop():
    global running

    running = False

    if wheels is not None:
        wheels.set_wheels_speed(0.0, 0.0)

    print("[Convoying] Stopped")
    return jsonify({'status': 'stopped'})


@app.route('/reset', methods=['POST'])
def reset():
    global running
    global _last_detections
    global _last_target
    global _last_valid_target
    global _last_command
    global _last_red_line_state

    running = False

    if wheels is not None:
        wheels.set_wheels_speed(0.0, 0.0)
        wheels.reset_game()

    if tracker is not None:
        tracker.reset()

    if red_line_gate is not None:
        red_line_gate.reset()

    with _detection_lock:
        _last_detections = []

    _last_target = None
    _last_valid_target = None
    _last_command = None
    _last_red_line_state = None

    print("[Convoying] Reset")
    return jsonify({'status': 'reset'})


@app.route('/running')
def get_running():
    return jsonify({'running': running})


@app.route('/status')
def status():
    game_over = wheels.is_game_over() if wheels is not None else False

    return jsonify({
        'running': running,
        'game_over': game_over,
        'model_loaded': bool(getattr(object_agent, 'model_loaded', False)) if object_agent else False,
        'model_load_error': getattr(object_agent, 'load_error', None) if object_agent else None,
        'trt_building': getattr(object_agent, 'trt_building', False) if object_agent else False,
        'lane_frame_count': getattr(lane_agent, 'frame_count', 0) if lane_agent else 0,
        'target': _target_to_dict(_last_target),
        'command': _command_to_dict(_last_command),
        'lane_left': _last_lane_left,
        'lane_right': _last_lane_right,
        'red_line': _red_line_to_dict(_last_red_line_state),
    })


@app.route('/update_config', methods=['POST'])
def update_config():
    """Tune convoy controller multipliers live from the browser."""
    data = request.json or {}

    if convoy_controller is not None:
        if 'close_multiplier' in data:
            convoy_controller.close_multiplier = float(data['close_multiplier'])
        if 'good_multiplier' in data:
            convoy_controller.good_multiplier = float(data['good_multiplier'])
        if 'far_multiplier' in data:
            convoy_controller.far_multiplier = float(data['far_multiplier'])
        if 'max_speed' in data:
            convoy_controller.max_speed = float(data['max_speed'])

        # Optional leader-only tuning from browser/API.
        if 'leader_steering_gain' in data:
            convoy_controller.leader_steering_gain = float(data['leader_steering_gain'])
        if 'leader_steering_sign' in data:
            convoy_controller.leader_steering_sign = float(data['leader_steering_sign'])

    return jsonify({
        'close_multiplier': convoy_controller.close_multiplier if convoy_controller else None,
        'good_multiplier': convoy_controller.good_multiplier if convoy_controller else None,
        'far_multiplier': convoy_controller.far_multiplier if convoy_controller else None,
        'max_speed': convoy_controller.max_speed if convoy_controller else None,
        'leader_steering_gain': convoy_controller.leader_steering_gain if convoy_controller else None,
        'leader_steering_sign': convoy_controller.leader_steering_sign if convoy_controller else None,
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_blue_truck_fallback(frame_rgb, detections, last_target=None):
    """
    Strict fallback detector for the blue lead truck in simulation.

    Main goals:
        - Do not convert signs into trucks.
        - Keep detection when the truck is near image edge during turns.
        - Keep box tight around blue truck body.
        - Prefer previous valid target if available.
    """
    if detections is None:
        detections = []

    # If YOLO already found a real truck, trust it.
    for _, _, class_id in detections:
        if class_id == 1:
            return detections

    h, w = frame_rgb.shape[:2]
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)

    # Blue truck body range.
    lower_blue = np.array([90, 40, 20])
    upper_blue = np.array([135, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Ignore sky/top area. Do NOT cut left/right too much because truck can be at edge in turns.
    mask[: int(h * 0.16), :] = 0

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sign_boxes = [bbox for bbox, _, class_id in detections if class_id == 2]

    previous_center = None
    if last_target is not None:
        if last_target.center_x is not None and last_target.center_y is not None:
            previous_center = (last_target.center_x, last_target.center_y)

    candidates = []

    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)

        if bw < 16 or bh < 20:
            continue

        # Tighten box around dense blue region.
        x, y, bw, bh = _tighten_blue_bbox(mask, x, y, bw, bh)

        bbox = (x, y, x + bw, y + bh)
        area = bw * bh
        bottom_y = y + bh
        center_x = x + bw / 2.0
        center_y = y + bh / 2.0

        if area < 700:
            continue

        # Signs are usually higher; truck body should be lower enough.
        if bottom_y < h * 0.28:
            continue

        # Reject huge merged boxes, but allow partial side-view on turns.
        aspect = bw / float(max(1, bh))
        if aspect > 1.45:
            continue

        # Reject blue blobs overlapping YOLO sign boxes.
        if any(_iou(bbox, sign_bbox) > 0.04 for sign_bbox in sign_boxes):
            continue

        # If we do not have previous target, avoid starting from extreme edge.
        # If we do have previous target, edge is allowed because turn can push truck sideways.
        if previous_center is None:
            if center_x < w * 0.08 or center_x > w * 0.92:
                continue

        if previous_center is not None:
            px, py = previous_center
            dist = ((center_x - px) ** 2 + (center_y - py) ** 2) ** 0.5
        else:
            dist = 0.0

        # Prefer lower/larger target. Previous target consistency helps but should not dominate.
        score = area + bottom_y * 80.0
        if previous_center is not None:
            score -= dist * 60.0

        candidates.append((score, bbox))

    if not candidates:
        return detections

    candidates.sort(reverse=True, key=lambda item: item[0])
    _, best_bbox = candidates[0]

    detections.append((best_bbox, 0.99, 1))
    return detections


def _tighten_blue_bbox(mask, x, y, bw, bh):
    """
    Crop a blue contour to the densest blue region.
    Prevents one wide box from covering truck + sign/road/background.
    """
    roi = mask[y:y + bh, x:x + bw]

    if roi.size == 0:
        return x, y, bw, bh

    col_counts = np.sum(roi > 0, axis=0)
    row_counts = np.sum(roi > 0, axis=1)

    if col_counts.max() <= 0 or row_counts.max() <= 0:
        return x, y, bw, bh

    col_threshold = max(3, int(col_counts.max() * 0.35))
    row_threshold = max(3, int(row_counts.max() * 0.20))

    valid_cols = np.where(col_counts >= col_threshold)[0]
    valid_rows = np.where(row_counts >= row_threshold)[0]

    if len(valid_cols) == 0 or len(valid_rows) == 0:
        return x, y, bw, bh

    new_x1 = int(valid_cols[0])
    new_x2 = int(valid_cols[-1])
    new_y1 = int(valid_rows[0])
    new_y2 = int(valid_rows[-1])

    new_x = x + new_x1
    new_y = y + new_y1
    new_w = max(1, new_x2 - new_x1)
    new_h = max(1, new_y2 - new_y1)

    return new_x, new_y, new_w, new_h


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    inter = iw * ih

    if inter <= 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return inter / union


def _target_to_dict(target):
    if target is None:
        return None

    return {
        'found': target.found,
        'bbox': target.bbox,
        'center_x': target.center_x,
        'center_y': target.center_y,
        'bottom_y': target.bottom_y,
        'area': target.area,
        'score': target.score,
        'class_id': target.class_id,
        'distance_state': target.distance_state,
        'reason': target.reason,
    }


def _command_to_dict(command):
    if command is None:
        return None

    return {
        'should_move': command.should_move,
        'left_speed': command.left_speed,
        'right_speed': command.right_speed,
        'speed_multiplier': command.speed_multiplier,
        'reason': command.reason,
    }


def _red_line_to_dict(red_line_state):
    if red_line_state is None:
        return None

    return {
        'red_line_close': red_line_state.red_line_close,
        'lane_disabled': red_line_state.lane_disabled,
        'disabled_remaining': red_line_state.disabled_remaining,
        'red_ratio': red_line_state.red_ratio,
        'red_row_ratio': red_line_state.red_row_ratio,
    }


def _placeholder(text):
    canvas = np.zeros((240, 640, 3), dtype='uint8')
    cv2.putText(
        canvas,
        text,
        (120, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (100, 100, 100),
        2,
    )
    return canvas


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global camera
    global wheels
    global object_agent
    global lane_agent
    global tracker
    global convoy_controller
    global red_line_gate

    ap = argparse.ArgumentParser(description="Virtual Convoying Server")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--frame-port", type=int, default=5001)
    ap.add_argument("--wheel-port", type=int, default=5002)
    ap.add_argument("--godot-host", type=str, default="localhost")
    args = ap.parse_args()

    suppress_http_logs()

    print("=" * 60)
    print("VIRTUAL CONVOYING SERVER")
    print("=" * 60)

    print("\n[1/5] Initializing wheels driver...")
    wheels = GodotWheelsDriver(
        WheelPWMConfiguration(pwm_min=0),
        WheelPWMConfiguration(pwm_min=0),
        godot_host=args.godot_host,
        godot_port=args.wheel_port,
    )
    wheels.trim = 0
    print(f"  Wheels: {args.godot_host}:{args.wheel_port}")

    print("\n[2/5] Initializing camera driver...")
    print(f"  Waiting for Godot on port {args.frame_port}...")
    camera = GodotCameraDriver(
        godot_config=GodotCameraConfig(host="0.0.0.0", port=args.frame_port)
    )
    camera.start()
    print("  Camera: connected!")

    print("\n[3/5] Loading object detection model...")
    object_agent = ObjectDetectionAgent()

    if object_agent.model_loaded:
        print(f"  Model ready ({object_agent.img_size}px)")
    elif getattr(object_agent, 'trt_building', False):
        print("  TensorRT engine building in background — will be ready shortly")
    else:
        print(f"  WARNING: {object_agent.load_error}")

    print("\n[4/5] Creating lane servoing agent...")
    lane_agent = LaneServoingAgent()
    print(f"  base_speed={lane_agent.base_speed}")

    print("\n[5/5] Creating convoying tracker, controller, and red-line gate...")
    tracker = TargetTracker()
    convoy_controller = ConvoyController()
    red_line_gate = RedLineGate(disable_seconds=5.0)

    print("  Tracker ready")
    print("  Controller ready")
    print("  Red-line gate ready: lane disabled for 5 seconds after close red line")

    threading.Thread(target=_detection_loop, daemon=True).start()

    web_port = find_available_port(args.port)

    if web_port != args.port:
        print(f"\n  Port {args.port} busy, using {web_port}")

    print("\n" + "=" * 60)
    print(f"Web Interface: http://localhost:{web_port}")
    print("=" * 60)
    print("\n  1. Make sure Godot sim is running")
    print("  2. Open the web interface")
    print("  3. Wait for 'Detection model loaded'")
    print("  4. Press Start")
    print("  5. Press Ctrl+C here to stop\n")
    print("=" * 60 + "\n")

    try:
        app.run(host='127.0.0.1', port=web_port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        shutdown_cleanup(wheels, camera, stop_event)


if __name__ == "__main__":
    sys.exit(main())