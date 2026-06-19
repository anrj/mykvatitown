"""
Project Server — SIMULATION (Godot convoying scene).

Mirror of servers/project/real_server.py but backed by the Godot
simulator. It honours the same project contract: it builds `camera`,
`wheels`, `leds`, `stop_event` and runs
`agent.main(camera, wheels, leds, stop_event)` on its own thread — so the
exact same tasks/project/packages/agent.py runs in sim and on the bot.

Extras for testing (dashboard at http://localhost:5000):
  * Auto / Manual toggle  — drive the follower with the arrow keys.
  * Reset                 — reload the scene (both bots back to start).

Manual mode is implemented without touching agent.py: the agent drives a
thin proxy (AgentWheels) whose commands are ignored while manual mode is
on, so the dashboard's manual loop has the wheels to itself.

Launched by `python launch.py --sim --task project`
(passes --port / --frame-port / --wheel-port).
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

import tasks.project.packages.agent as agent

_SIM_CONFIG_FILE = 'project_config_sim.yaml'
agent.CONFIG_FILE = _SIM_CONFIG_FILE          # must be set before agent.main() is called
CONFIG_PATH = os.path.join(project_root, 'config', _SIM_CONFIG_FILE)

# Slider schema: (section, key, label, min, max, step)
_CONFIG_SLIDERS = [
    ('leader',  'target_span',   'Target Span',      0.05, 0.80, 0.01),
    ('leader',  'stop_span',     'Stop Span',        0.10, 0.90, 0.01),
    ('leader',  'span_deadband', 'Span Deadband',    0.00, 0.10, 0.005),
    ('control', 'max_speed',     'Max Speed',        0.00, 1.00, 0.01),
    ('control', 'chase_speed',   'Chase Speed',      0.00, 0.60, 0.01),
    ('control', 'steer_kp',      'Steer Kp',         0.00, 2.00, 0.01),
    ('control', 'steer_kd',      'Steer Kd',         0.00, 1.00, 0.01),
    ('control', 'dist_kp',       'Dist Kp',          0.00, 5.00, 0.01),
    ('control', 'accel_rate',    'Accel Rate',       0.00, 0.20, 0.01),
    ('control', 'decel_rate',    'Decel Rate',       0.00, 0.20, 0.01),
    ('control', 'search_turn',   'Search Turn',      0.00, 0.40, 0.01),
    ('signs',   'min_tag_px',    'Min Tag Size (px)', 10,   100,   1),
    ('signs',   'stop_hold_s',   'Stop Hold (s)',    0.50, 5.00, 0.10),
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
wheels     = None              # real GodotWheelsDriver (server owns it)
leds       = None
stop_event = threading.Event()

MANUAL_MODE = False
_keys = {'up': False, 'down': False, 'left': False, 'right': False}
_keys_lock = threading.Lock()
_keys_stamp = 0.0


# --- wheel proxy: agent's commands are dropped while driving manually ----
class AgentWheels:
    def __init__(self, inner):
        self._inner = inner

    def set_wheels_speed(self, left, right):
        if not MANUAL_MODE:
            self._inner.set_wheels_speed(left, right)

    def __getattr__(self, name):       # delegate everything else to the real driver
        return getattr(self._inner, name)


def _manual_loop():
    """Drive the follower from the arrow keys while in manual mode."""
    global _keys_stamp
    while not stop_event.is_set():
        if not MANUAL_MODE:
            time.sleep(0.05)
            continue
        # auto-release keys if the browser stopped sending (safety)
        if time.time() - _keys_stamp > 0.5:
            with _keys_lock:
                for k in _keys:
                    _keys[k] = False
        with _keys_lock:
            k = dict(_keys)

        left = right = 0.0
        if k['up']:
            left = right = 0.5
        elif k['down']:
            left = right = -0.4
        if k['left']:
            left, right = left - 0.3, right + 0.3
        elif k['right']:
            left, right = left + 0.3, right - 0.3
        wheels.set_wheels_speed(max(-1, min(1, left)), max(-1, min(1, right)))
        time.sleep(0.05)


def _visualize(frame):
    """Always show the agent's annotated detection view (dots / state)."""
    debug = getattr(agent, 'DEBUG_FRAME', None)
    if debug is not None:
        return debug
    if frame is not None:
        return frame
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(blank, "Waiting for camera...", (160, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
    return blank


# GodotCameraDriver.read() returns BGR (rgb=False) and is concurrent-safe, so the
# agent and this feed both read it directly — no queue, no extra reader thread.
generate_frames = make_frame_generator(lambda: camera, _visualize, quality=70, rgb=False)


@app.route('/')
def index():
    return render_template_string(_HTML)


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    st = dict(getattr(agent, 'STATUS', {}) or {})
    st['mode'] = 'manual' if MANUAL_MODE else 'auto'
    return jsonify(st)


@app.route('/set_mode', methods=['POST'])
def set_mode():
    global MANUAL_MODE
    mode = (request.json or {}).get('mode', 'auto')
    MANUAL_MODE = (mode == 'manual')
    if wheels:
        wheels.set_wheels_speed(0.0, 0.0)      # neutral on every switch
    return jsonify({'mode': 'manual' if MANUAL_MODE else 'auto'})


@app.route('/keys', methods=['POST'])
def keys():
    global _keys_stamp
    data = request.json or {}
    with _keys_lock:
        for k in _keys:
            _keys[k] = bool(data.get(k, False))
    _keys_stamp = time.time()
    return jsonify({'status': 'ok'})


@app.route('/reset', methods=['POST'])
def reset():
    # Reload the scene so BOTH bots return to the start (the leader's
    # progress resets too; a plain reset_game only resets the follower).
    if wheels:
        wheels.change_scene(GODOT_SCENES['project'])
    return jsonify({'status': 'reset'})


@app.route('/start_leader', methods=['POST'])
def start_leader():
    if wheels:
        wheels.start_leader()
    return jsonify({'status': 'ok'})


@app.route('/get_config')
def get_config():
    cfg = getattr(agent, 'CFG', None) or _load_config_file()
    return jsonify(cfg)


@app.route('/update_config', methods=['POST'])
def update_config():
    data = request.json or {}
    for section, values in data.items():
        if isinstance(values, dict):
            if section == 'tag_meanings':
                continue
            if agent.CFG is not None:
                agent.CFG.setdefault(section, {}).update(values)
    # persist to the YAML file
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

    ap = argparse.ArgumentParser(description='Project Server — Simulation')
    ap.add_argument('--port',       type=int, default=5000)
    ap.add_argument('--frame-port', type=int, default=5001)
    ap.add_argument('--wheel-port', type=int, default=5002)
    ap.add_argument('--godot-host', type=str, default='localhost')
    args = ap.parse_args()

    suppress_http_logs()
    print('=' * 60)
    print('PROJECT SERVER — SIMULATION (convoying)')
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

    print('\n[4/4] Starting agent...')
    stop_event.clear()
    threading.Thread(
        target=agent.main,
        args=(camera, AgentWheels(wheels), leds, stop_event),
        daemon=True, name='AgentThread',
    ).start()
    threading.Thread(target=_manual_loop, daemon=True, name='ManualLoop').start()
    print('  agent.main() running')

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


# --- full-viewport dashboard: video + status + controls + config sliders ---
_SLIDERS_JSON = json.dumps(_CONFIG_SLIDERS)
_HTML = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Convoying — Sim</title>
<style>
 *{{margin:0;padding:0;box-sizing:border-box}}
 :root{{--bg:#13161a;--surface:#1a1d23;--border:#30363d;--text:#e6edf3;
       --muted:#8b949e;--accent:#1f6feb;--accent-hover:#388bfd;--danger:#7d2622}}
 body{{background:var(--bg);color:var(--text);font-family:Inter,sans-serif;
       height:100vh;overflow:hidden;display:flex;flex-direction:column}}
 .header{{display:flex;align-items:center;gap:12px;padding:10px 16px;
          border-bottom:1px solid var(--border);flex-shrink:0}}
 .header h1{{font-size:18px;font-weight:600}}
 .header .sub{{color:var(--muted);font-size:13px}}
 .main{{display:grid;grid-template-columns:1fr 420px;flex:1;min-height:0}}
 .video-wrap{{background:#000;display:flex;align-items:center;justify-content:center;
              position:relative;overflow:hidden}}
 .video-wrap img{{width:100%;height:100%;object-fit:contain;display:block}}
 .sidebar{{display:flex;flex-direction:column;gap:8px;padding:10px 12px 12px;
           overflow-y:auto;border-left:1px solid var(--border);background:var(--bg)}}
 .card{{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:12px}}
 .card-h{{font-size:13px;font-weight:600;margin-bottom:10px;padding-bottom:6px;
          border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center}}
 .row{{display:flex;justify-content:space-between;padding:3px 0;font-size:12px}}
 .row+.row{{border-top:1px solid #222}}
 .k{{color:var(--muted)}}.v{{font-family:monospace;color:var(--text)}}
 button{{background:#21262d;color:var(--text);border:1px solid var(--border);border-radius:6px;
        padding:8px 12px;font-size:13px;cursor:pointer;transition:.15s;font-family:inherit}}
 button:hover{{background:#30363d}}
 button.on{{background:var(--accent);border-color:var(--accent)}}
 button.danger{{background:var(--danger);border-color:#a13}}
 button.danger:hover{{background:#8d2d28}}
 .mode-row{{display:flex;gap:6px}}
 .mode-row button{{flex:1}}
 .arrow-pad{{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;margin-top:8px}}
 .arrow-pad button{{height:40px;font-size:16px;padding:0}}
 .arrow-pad .sp{{visibility:hidden}}
 .sg{{margin-bottom:14px}}.sg:last-child{{margin-bottom:0}}
 .sg-t{{font-size:11px;font-weight:600;color:var(--muted);text-transform:uppercase;
        letter-spacing:.5px;margin-bottom:8px;padding-bottom:4px;border-bottom:1px solid var(--border)}}
 .sl{{display:flex;justify-content:space-between;margin-bottom:2px;font-size:11px;color:var(--muted)}}
 .sc{{display:flex;gap:6px;align-items:center}}
 .s{{flex:1;height:4px;background:#0d1117;outline:none;border-radius:2px;appearance:none}}
 .s::-webkit-slider-thumb{{appearance:none;width:13px;height:13px;background:var(--accent);
                           cursor:pointer;border-radius:50%;border:2px solid var(--surface)}}
 .sv{{width:52px;padding:3px 4px;background:#0d1117;border:1px solid var(--border);
      border-radius:4px;color:var(--text);font-size:11px;text-align:center;font-family:monospace}}
 .sv:focus{{outline:none;border-color:var(--accent)}}
</style></head><body>
<div class="header">
  <h1>Convoying — Simulation</h1>
  <span class="sub">Leader Following · PID Tuning</span>
</div>
<div class="main">
  <div class="video-wrap">
    <img id="feed" src="/video">
  </div>
  <div class="sidebar" id="sidebar">
    <!-- status -->
    <div class="card">
      <div class="card-h">Status <span id="dot" style="width:7px;height:7px;border-radius:50%;
          background:var(--accent);display:inline-block"></span></div>
      <div id="status"></div>
    </div>
    <!-- mode + controls -->
    <div class="card">
      <div class="card-h">Controls</div>
      <div class="mode-row">
        <button id="autoBtn" class="on" onclick="setMode('auto')">Auto</button>
        <button id="manBtn" onclick="setMode('manual')">Manual</button>
      </div>
      <div class="arrow-pad">
        <div class="sp"></div><button onmousedown="press('up',1)" onmouseup="press('up',0)" onmouseleave="press('up',0)">▲</button><div class="sp"></div>
        <button onmousedown="press('left',1)" onmouseup="press('left',0)" onmouseleave="press('left',0)">◀</button>
        <button onmousedown="press('down',1)" onmouseup="press('down',0)" onmouseleave="press('down',0)">▼</button>
        <button onmousedown="press('right',1)" onmouseup="press('right',0)" onmouseleave="press('right',0)">▶</button>
      </div>
      <p style="color:var(--muted);font-size:11px;margin:6px 0 0">Arrow keys / WASD in manual mode.</p>
      <div style="display:flex;gap:6px;margin-top:8px">
        <button style="flex:1" onclick="startLeader()">▶ Start Leader</button>
        <button class="danger" style="flex:1" onclick="doReset()">↺ Reset</button>
      </div>
    </div>
    <!-- config sliders -->
    <div class="card">
      <div class="card-h">Leader Config</div>
      <div id="sliders-leader"></div>
    </div>
    <div class="card">
      <div class="card-h">Control Config</div>
      <div id="sliders-control"></div>
    </div>
    <div class="card">
      <div class="card-h">Sign Config</div>
      <div id="sliders-signs"></div>
    </div>
  </div>
</div>
<script>
const SLIDERS={_SLIDERS_JSON};
let keys={{up:false,down:false,left:false,right:false}};
function post(u,b){{return fetch(u,{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(b||{{}})}});}}
function sendKeys(){{post('/keys',keys);}}
function press(k,on){{keys[k]=!!on;sendKeys();}}
function doReset(){{post('/reset',{{}});}}
function startLeader(){{post('/start_leader',{{}});}}
function setMode(m){{post('/set_mode',{{mode:m}}).then(()=>{{
  document.getElementById('autoBtn').className=m==='auto'?'on':'';
  document.getElementById('manBtn').className=m==='manual'?'on':'';}});}}
const KMAP={{ArrowUp:'up',ArrowDown:'down',ArrowLeft:'left',ArrowRight:'right',w:'up',s:'down',a:'left',d:'right'}};
addEventListener('keydown',e=>{{if(KMAP[e.key]&&!keys[KMAP[e.key]]){{keys[KMAP[e.key]]=true;sendKeys();e.preventDefault();}}}});
addEventListener('keyup',e=>{{if(KMAP[e.key]){{keys[KMAP[e.key]]=false;sendKeys();e.preventDefault();}}}});

function buildSliders(){{
  const secs={{}};
  SLIDERS.forEach(s=>{{
    const sec=s[0]; if(!secs[sec])secs[sec]=[];
    secs[sec].push(s);
  }});
  Object.keys(secs).forEach(sec=>{{
    const el=document.getElementById('sliders-'+sec);
    if(!el)return;
    el.innerHTML=secs[sec].map(s=>{{
      const k=s[1],label=s[2],min=s[3],max=s[4],step=s[5];
      return '<div class="sg"><div class="sl"><span>'+label+'</span><span class="sv" id="dv-'+k+'">—</span></div>'+
        '<div class="sc"><input type="range" class="s" id="sl-'+k+'" min="'+min+'" max="'+max+'" step="'+step+'">'+
        '<input type="number" class="sv" id="in-'+k+'" min="'+min+'" max="'+max+'" step="'+step+'"></div></div>';
    }}).join('');
  }});
}}

function loadConfig(){{
  fetch('/get_config').then(r=>r.json()).then(cfg=>{{
    SLIDERS.forEach(s=>{{
      const sec=s[0],k=s[1];
      const v=(((cfg[sec]||{{}})[k])!==undefined?cfg[sec][k]:s[3]);
      const sl=document.getElementById('sl-'+k);
      const inp=document.getElementById('in-'+k);
      const dv=document.getElementById('dv-'+k);
      if(sl){{sl.value=v;if(dv)dv.textContent=parseFloat(v).toFixed(2)}}
      if(inp)inp.value=v;
    }});
  }}).catch(()=>{{}});
}}

let timeouts={{}};
function sliderChanged(k){{
  const sl=document.getElementById('sl-'+k);
  const inp=document.getElementById('in-'+k);
  const dv=document.getElementById('dv-'+k);
  const v=sl?parseFloat(sl.value):0;
  if(inp)inp.value=v;
  if(dv)dv.textContent=v.toFixed(2);
  clearTimeout(timeouts[k]);
  timeouts[k]=setTimeout(()=>{{
    for(let i=0;i<SLIDERS.length;i++){{
      if(SLIDERS[i][1]===k){{
        const sec=SLIDERS[i][0];
        const body={{}};body[sec]={{}};body[sec][k]=v;
        post('/update_config',body);
        break;
      }}
    }}
  }},300);
}}

function initSliders(){{
  SLIDERS.forEach(s=>{{
    const k=s[1];
    const sl=document.getElementById('sl-'+k);
    const inp=document.getElementById('in-'+k);
    if(sl)sl.addEventListener('input',()=>sliderChanged(k));
    if(inp)inp.addEventListener('input',()=>{{
      let v=parseFloat(inp.value);
      if(isNaN(v))return;
      v=Math.max(parseFloat(inp.min),Math.min(parseFloat(inp.max),v));
      const sl2=document.getElementById('sl-'+k);
      if(sl2)sl2.value=v;
      sliderChanged(k);
    }});
  }});
}}

buildSliders();loadConfig();initSliders();

setInterval(()=>{{
  fetch('/status').then(r=>r.json()).then(d=>{{
    const dot=document.getElementById('dot');
    dot.style.background=d.mode==='manual'?'var(--accent)':'#3fb950';
    document.getElementById('status').innerHTML=Object.entries(d).map(
      ([k,v])=>'<div class="row"><span class="k">'+k+'</span><span class="v">'+JSON.stringify(v)+'</span></div>'
    ).join('');
  }}).catch(()=>{{}});
}},400);
</script></body></html>"""


if __name__ == '__main__':
    sys.exit(main())
