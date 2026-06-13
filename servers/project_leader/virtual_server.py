"""
Project Leader Server — SIMULATION.

Tests lane follow + stop-sign detection on the player Duckiebot.

Launched by `python launch.py --sim --task project_leader`
"""

import sys
import os
import json
import signal
import threading
import time
import argparse

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..')
sys.path.insert(0, project_root)

from flask import Flask, Response, jsonify, request, render_template_string
import numpy as np
import cv2
import yaml

from duckiebot.camera_driver.godot_camera_driver import GodotCameraDriver, GodotCameraConfig
from duckiebot.wheel_driver.godot_wheels_driver import GodotWheelsDriver
from duckiebot.wheel_driver.wheels_driver_abs import WheelPWMConfiguration
from duckiebot.led_driver import VirtualLEDsDriver
from launcher.ports import find_available_port
from launcher.config import GODOT_SCENES
from servers.common import make_frame_generator, shutdown_cleanup, suppress_http_logs
from servers.manual_control import ManualDriveController, MANUAL_KEY_JS, MANUAL_PAD_HTML, MANUAL_PAD_CSS

import tasks.project_leader.packages.leader_agent as leader_agent

_SIM_CONFIG_FILE = 'leader_config_sim.yaml'
leader_agent.CONFIG_FILE = _SIM_CONFIG_FILE
CONFIG_PATH = os.path.join(project_root, 'config', _SIM_CONFIG_FILE)

_CONFIG_SLIDERS = [
    ('lane_stop', 'stop_hold_s',        'Stop Hold (s)',       0.5, 3.0, 0.1),
    ('lane_stop', 'line_speed_mult',    'White-Line Speed',    0.1, 0.6, 0.01),
    ('lane_stop', 'line_lost_frames', 'Line Lost Frames', 2, 12, 1),
]


def _load_config_file():
    try:
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _save_config_file(data):
    try:
        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    except Exception as e:
        print(f'[server] Could not save config: {e}')


app        = Flask(__name__)
camera     = None
wheels     = None
leds       = None
stop_event = threading.Event()
manual = ManualDriveController()


class AgentWheels:
    def __init__(self, inner):
        self._inner = inner

    def set_wheels_speed(self, left, right):
        if not manual.manual_mode:
            self._inner.set_wheels_speed(left, right)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _visualize(frame):
    debug = getattr(leader_agent, 'DEBUG_FRAME', None)
    if debug is not None:
        return debug
    if frame is not None:
        return frame
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(blank, 'Waiting for camera...', (160, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
    return blank


generate_frames = make_frame_generator(lambda: camera, _visualize, quality=70, rgb=True)


@app.route('/')
def index():
    return render_template_string(_HTML)


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    st = dict(getattr(leader_agent, 'STATUS', {}) or {})
    st['mode'] = 'manual' if manual.manual_mode else 'auto'
    return jsonify(st)


@app.route('/set_mode', methods=['POST'])
def set_mode():
    mode = (request.json or {}).get('mode', 'auto')
    manual.set_mode(mode)
    if wheels:
        wheels.set_wheels_speed(0.0, 0.0)
    return jsonify({'mode': 'manual' if manual.manual_mode else 'auto'})


@app.route('/keys', methods=['POST'])
def keys():
    manual.update_keys(request.json or {})
    return jsonify({'status': 'ok'})


@app.route('/reset', methods=['POST'])
def reset():
    if wheels:
        wheels.change_scene(GODOT_SCENES['project_leader'])
    return jsonify({'status': 'reset'})


@app.route('/get_config')
def get_config():
    cfg = getattr(leader_agent, 'CFG', None) or _load_config_file()
    return jsonify(cfg)


@app.route('/update_config', methods=['POST'])
def update_config():
    data = request.json or {}
    for section, values in data.items():
        if isinstance(values, dict):
            if section == 'tag_meanings':
                continue
            if leader_agent.CFG is not None:
                leader_agent.CFG.setdefault(section, {}).update(values)
    on_disk = _load_config_file()
    for section, values in data.items():
        if isinstance(values, dict):
            on_disk.setdefault(section, {}).update(values)
    _save_config_file(on_disk)
    return jsonify({'status': 'ok'})


@app.route('/shutdown')
def shutdown():
    shutdown_cleanup(wheels, camera, stop_event)
    return jsonify({'status': 'ok'})


def main():
    global camera, wheels, leds, stop_event

    ap = argparse.ArgumentParser(description='Project Leader Server — Simulation')
    ap.add_argument('--port',       type=int, default=5000)
    ap.add_argument('--frame-port', type=int, default=5001)
    ap.add_argument('--wheel-port', type=int, default=5002)
    ap.add_argument('--godot-host', type=str, default='localhost')
    args = ap.parse_args()

    suppress_http_logs()
    print('=' * 60)
    print('PROJECT LEADER SERVER — SIMULATION')
    print('=' * 60)

    print('\n[1/4] Initializing virtual LEDs...')
    leds = VirtualLEDsDriver(debug=False)
    leds.all_off()

    print('\n[2/4] Initializing wheels (Godot)...')
    wheels = GodotWheelsDriver(
        WheelPWMConfiguration(pwm_min=0), WheelPWMConfiguration(pwm_min=0),
        godot_host=args.godot_host, godot_port=args.wheel_port,
    )

    print('\n[3/4] Initializing camera (Godot)...')
    camera = GodotCameraDriver(godot_config=GodotCameraConfig(host='0.0.0.0', port=args.frame_port))
    camera.start()

    print('\n[4/4] Starting leader agent...')
    stop_event.clear()
    threading.Thread(
        target=leader_agent.main,
        args=(camera, AgentWheels(wheels), leds, stop_event),
        daemon=True, name='LeaderAgentThread',
    ).start()
    threading.Thread(
        target=manual.run_loop,
        args=(stop_event, wheels.set_wheels_speed),
        daemon=True, name='ManualLoop',
    ).start()
    print('  leader_agent.main() running')

    def _shutdown(signum, frame):
        print('\nShutting down...')
        shutdown_cleanup(wheels, camera, stop_event)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    web_port = find_available_port(args.port)
    print(f'\nWeb Interface: http://localhost:{web_port}')
    print('=' * 60 + '\n')

    try:
        app.run(host='127.0.0.1', port=web_port, debug=False, threaded=True)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        shutdown_cleanup(wheels, camera, stop_event)


_SLIDERS_JSON = json.dumps(_CONFIG_SLIDERS)
_HTML = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Convoy Leader — Sim</title>
<style>
 *{{margin:0;padding:0;box-sizing:border-box}}
 :root{{--bg:#13161a;--surface:#1a1d23;--border:#30363d;--text:#e6edf3;
       --muted:#8b949e;--accent:#1f6feb;--danger:#7d2622}}
 body{{background:var(--bg);color:var(--text);font-family:Inter,sans-serif;
       height:100vh;overflow:hidden;display:flex;flex-direction:column}}
 .header{{display:flex;align-items:center;gap:12px;padding:10px 16px;border-bottom:1px solid var(--border)}}
 .header h1{{font-size:18px}} .header .sub{{color:var(--muted);font-size:13px}}
 .main{{display:grid;grid-template-columns:1fr 380px;flex:1;min-height:0}}
 .video-wrap{{background:#000;display:flex;align-items:center;justify-content:center;overflow:hidden}}
 .video-wrap img{{width:100%;height:100%;object-fit:contain}}
 .sidebar{{padding:10px 12px;overflow-y:auto;border-left:1px solid var(--border)}}
 .card{{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:12px;margin-bottom:8px}}
 .card-h{{font-size:13px;font-weight:600;margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid var(--border)}}
 .row{{display:flex;justify-content:space-between;padding:3px 0;font-size:12px}}
 .k{{color:var(--muted)}}.v{{font-family:monospace}}
 button{{background:#21262d;color:var(--text);border:1px solid var(--border);border-radius:6px;
        padding:8px 12px;font-size:13px;cursor:pointer;font-family:inherit}}
 button.on{{background:var(--accent);border-color:var(--accent)}}
 button.danger{{background:var(--danger)}}
 .mode-row{{display:flex;gap:6px}} .mode-row button{{flex:1}}
{MANUAL_PAD_CSS}
 .sg{{margin-bottom:12px}} .sl{{display:flex;justify-content:space-between;font-size:11px;color:var(--muted);margin-bottom:2px}}
 .sc{{display:flex;gap:6px;align-items:center}}
 .s{{flex:1;height:4px;background:#0d1117;appearance:none;border-radius:2px}}
 .s::-webkit-slider-thumb{{appearance:none;width:13px;height:13px;background:var(--accent);border-radius:50%}}
 .sv{{width:52px;padding:3px;background:#0d1117;border:1px solid var(--border);border-radius:4px;
      color:var(--text);font-size:11px;text-align:center;font-family:monospace}}
</style></head><body>
<div class="header">
  <h1>Convoy Leader — Simulation</h1>
  <span class="sub">Lane Follow · Sign Stop</span>
</div>
<div class="main">
  <div class="video-wrap"><img id="feed" src="/video"></div>
  <div class="sidebar">
    <div class="card"><div class="card-h">Status</div><div id="status"></div></div>
    <div class="card">
      <div class="card-h">Controls</div>
      <div class="mode-row">
        <button id="autoBtn" class="on" onclick="setMode('auto')">Auto</button>
        <button id="manBtn" onclick="setMode('manual')">Manual</button>
      </div>
      <button class="danger" style="width:100%;margin-top:8px" onclick="post('/reset',{{}})">↺ Reset</button>
      {MANUAL_PAD_HTML}
    </div>
    <div class="card"><div class="card-h">Lane Stop Config</div><div id="sliders-lane_stop"></div></div>
  </div>
</div>
<script>
const SLIDERS={_SLIDERS_JSON};
function post(u,b){{return fetch(u,{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(b||{{}})}});}}
function setMode(m){{post('/set_mode',{{mode:m}}).then(()=>{{
  document.getElementById('autoBtn').className=m==='auto'?'on':'';
  document.getElementById('manBtn').className=m==='manual'?'on':'';}});}}
{MANUAL_KEY_JS}
function buildSliders(){{
  document.getElementById('sliders-lane_stop').innerHTML=SLIDERS.map(s=>{{
    const k=s[1],label=s[2],min=s[3],max=s[4],step=s[5];
    return '<div class="sg"><div class="sl"><span>'+label+'</span><span id="dv-'+k+'">—</span></div>'+
      '<div class="sc"><input type="range" class="s" id="sl-'+k+'" min="'+min+'" max="'+max+'" step="'+step+'">'+
      '<input type="number" class="sv" id="in-'+k+'" min="'+min+'" max="'+max+'" step="'+step+'"></div></div>';
  }}).join('');
}}
function loadConfig(){{
  fetch('/get_config').then(r=>r.json()).then(cfg=>{{
    SLIDERS.forEach(s=>{{
      const k=s[1],v=(((cfg[s[0]]||{{}})[k])!==undefined?cfg[s[0]][k]:s[3]);
      const sl=document.getElementById('sl-'+k),inp=document.getElementById('in-'+k),dv=document.getElementById('dv-'+k);
      if(sl){{sl.value=v;if(dv)dv.textContent=parseFloat(v).toFixed(2)}}
      if(inp)inp.value=v;
    }});
  }});
}}
let timeouts={{}};
function sliderChanged(k){{
  const sl=document.getElementById('sl-'+k),dv=document.getElementById('dv-'+k);
  const v=sl?parseFloat(sl.value):0;
  if(dv)dv.textContent=v.toFixed(2);
  clearTimeout(timeouts[k]);
  timeouts[k]=setTimeout(()=>{{
    SLIDERS.forEach(s=>{{if(s[1]===k){{const body={{}};body[s[0]]={{}};body[s[0]][k]=v;post('/update_config',body);}}}});
  }},300);
}}
buildSliders();loadConfig();
SLIDERS.forEach(s=>{{
  const k=s[1],sl=document.getElementById('sl-'+k);
  if(sl)sl.addEventListener('input',()=>sliderChanged(k));
}});
setInterval(()=>{{
  fetch('/status').then(r=>r.json()).then(d=>{{
    document.getElementById('status').innerHTML=Object.entries(d).map(
      ([k,v])=>'<div class="row"><span class="k">'+k+'</span><span class="v">'+JSON.stringify(v)+'</span></div>'
    ).join('');
  }}).catch(()=>{{}});
}},400);
</script></body></html>"""


if __name__ == '__main__':
    sys.exit(main())
