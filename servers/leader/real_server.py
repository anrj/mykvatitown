"""
Leader Server — REAL HARDWARE.

Launched by `python launch.py --run --bot <leader-hostname> --task leader`
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

from duckiebot.camera_driver import CameraDriver
from duckiebot.wheel_driver import DaguWheelsDriver
from duckiebot.wheel_driver.wheels_driver_abs import WheelPWMConfiguration
from duckiebot.led_driver import LEDDriver
from launcher.ports import find_available_port
from servers.common import make_frame_generator, shutdown_cleanup, suppress_http_logs

import tasks.leader.packages.agent as agent

CONFIG_PATH = os.path.join(project_root, 'config', agent.CONFIG_FILE)

_CONFIG_SLIDERS = [
    ('red_line', 'red_pixel_frac',        'Red Pixel Frac',       0.001, 0.05,  0.001),
    ('red_line', 'red_approach_pixel_frac','Approach Pixel Frac', 0.001, 0.02,  0.001),
    ('red_line', 'red_min_area',          'Red Min Area',         10,    300,   5),
    ('turn',     'stop_hold_s',           'Stop Hold (s)',        0.5,   5.0,   0.1),
    ('turn',     'turn_right_s',          'Turn Right (s)',       0.3,   3.0,   0.05),
    ('turn',     'turn_left_s',           'Turn Left (s)',        0.3,   4.0,   0.05),
    ('turn',     'approach_slow_factor',  'Approach Slow',        0.5,   0.99,  0.01),
    ('turn',     'turn_cooldown_s',       'Turn Cooldown (s)',    1.0,   10.0,  0.5),
    ('led',      'blink_hz',              'LED Blink Hz',         1.0,   10.0,  0.5),
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

_latest_frame = None
_latest_lock  = threading.Lock()

# Self-healing state for the camera loop. Without try/except around
# camera.read(), a single transient nvarguscamerasrc/GStreamer exception
# silently kills the only producer thread -> the dashboard freezes forever
# on a live-looking stale frame (the known Jetson IMX219 blip). The loop
# below mirrors make_frame_generator's try/except self-heal and additionally
# re-opens the camera if no fresh frame arrives for _RECONNECT_TIMEOUT.
_last_good_frame_time  = 0.0
_consecutive_failures  = 0
_RECONNECT_TIMEOUT     = 2.0


def _reconnect_camera():
    """Tear down and re-open the GStreamer pipeline (best effort)."""
    try:
        camera.stop()
    except Exception as e:
        print(f"[CameraLoop] stop() during reconnect: {e}")
    try:
        camera.start()
        print("[CameraLoop] Camera reconnected.")
    except Exception as e:
        print(f"[CameraLoop] Reconnect failed: {e}")
    global _last_good_frame_time, _consecutive_failures
    _last_good_frame_time = time.time()
    _consecutive_failures = 0


def _camera_loop():
    """Continuously pull frames from the real camera into _latest_frame.

    Self-heals transient read() exceptions and reconnects the camera if it
    stops producing frames. The agent and /video feed read _latest_frame
    (non-consuming), so a producer hiccup is masked rather than permanent."""
    global _latest_frame, _last_good_frame_time, _consecutive_failures
    _last_good_frame_time = time.time()
    while not stop_event.is_set():
        try:
            ok, frame = camera.read()
        except Exception as e:
            print(f"[CameraLoop] read() exception: {e}")
            ok, frame = False, None

        if ok and frame is not None:
            with _latest_lock:
                _latest_frame = frame
            _last_good_frame_time = time.time()
            _consecutive_failures = 0
        else:
            _consecutive_failures += 1
            time.sleep(0.01 if _consecutive_failures < 10 else 0.1)

        if not stop_event.is_set() and (time.time() - _last_good_frame_time) > _RECONNECT_TIMEOUT:
            print(f"[CameraLoop] No fresh frames for >{_RECONNECT_TIMEOUT:.1f}s — reconnecting...")
            _reconnect_camera()


def _run_agent_supervised(frame_source, wheels, leds, stop_event):
    """Run agent.main, restarting it if it dies unexpectedly. Same self-heal
    idea as _camera_loop: a transient per-frame exception in agent.main must
    not permanently freeze the dashboard overlay (which reads agent.DETECTION)."""
    while not stop_event.is_set():
        try:
            agent.main(frame_source, AgentWheels(wheels), leds, stop_event)
            return  # main returned cleanly (shutdown)
        except Exception as e:
            if stop_event.is_set():
                return
            print(f"[AgentThread] agent.main crashed: {e}; restarting in 1s...")
            stop_event.wait(1.0)


class _LiveFrameSource:
    def read(self):
        with _latest_lock:
            f = _latest_frame
        if f is None:
            return False, None
        return True, f


class _FrameSource:
    def read(self):
        with _latest_lock:
            f = _latest_frame
        if f is None:
            return False, None
        return True, f.copy()


_frame_source = _FrameSource()
_live_source = _LiveFrameSource()


MANUAL_MODE = False
_keys = {'up': False, 'down': False, 'left': False, 'right': False}
_keys_lock  = threading.Lock()
_keys_stamp = 0.0


class AgentWheels:
    def __init__(self, inner):
        self._inner = inner

    def set_wheels_speed(self, left, right):
        if not MANUAL_MODE:
            self._inner.set_wheels_speed(left, right)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _manual_loop():
    global _keys_stamp
    while not stop_event.is_set():
        if not MANUAL_MODE or getattr(agent, 'PAUSED', True):
            time.sleep(0.05)
            continue
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


_BLANK = np.zeros((480, 640, 3), dtype=np.uint8)


def _visualize_overlay(frame):
    det = {}
    with agent._det_lock:
        det = dict(agent.DETECTION)
    if not det:
        return frame if frame is not None else _BLANK.copy()
    return agent._annotate(frame, det)


def _visualize_raw(frame):
    if frame is not None:
        return frame
    img = _BLANK.copy()
    cv2.putText(img, "Waiting for camera...", (160, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
    return img


generate_frames     = make_frame_generator(lambda: _live_source, _visualize_overlay, quality=70, rgb=False)
generate_raw_frames = make_frame_generator(lambda: _live_source, _visualize_raw,     quality=70, rgb=False)


@app.route('/')
def index():
    return render_template_string(_HTML)


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/raw')
def raw_video():
    return Response(generate_raw_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    st = dict(getattr(agent, 'STATUS', {}) or {})
    st['mode'] = 'manual' if MANUAL_MODE else 'auto'
    st['paused'] = getattr(agent, 'PAUSED', True)
    return jsonify(st)


@app.route('/start', methods=['POST'])
def start():
    agent.PAUSED = False
    return jsonify({'paused': False})


@app.route('/stop', methods=['POST'])
def stop():
    agent.PAUSED = True
    if wheels:
        wheels.set_wheels_speed(0.0, 0.0)
    return jsonify({'paused': True})


@app.route('/set_mode', methods=['POST'])
def set_mode():
    global MANUAL_MODE
    mode = (request.json or {}).get('mode', 'auto')
    MANUAL_MODE = (mode == 'manual')
    if wheels:
        wheels.set_wheels_speed(0.0, 0.0)
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


@app.route('/get_config')
def get_config():
    cfg = getattr(agent, 'CFG', None) or _load_config_file()
    return jsonify(cfg)


@app.route('/update_config', methods=['POST'])
def update_config():
    data = request.json or {}
    for section, values in data.items():
        if isinstance(values, dict):
            if agent.CFG is not None:
                agent.CFG.setdefault(section, {}).update(values)
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

    ap = argparse.ArgumentParser(description='Leader Server — Real Hardware')
    ap.add_argument('--port', type=int, default=5000)
    args = ap.parse_args()

    suppress_http_logs()
    print('=' * 60)
    print('LEADER SERVER — REAL HARDWARE')
    print('=' * 60)

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
    print('  Wheels: ok')

    print('\n[3/4] Initializing camera driver...')
    camera = CameraDriver()
    camera.start()
    print('  Camera: ok')

    print('\n[4/4] Starting leader agent...')
    stop_event.clear()
    threading.Thread(target=_camera_loop, daemon=True, name='CameraLoop').start()
    threading.Thread(
        target=_run_agent_supervised,
        args=(_frame_source, AgentWheels(wheels), leds, stop_event),
        daemon=True, name='LeaderAgentThread',
    ).start()
    threading.Thread(target=_manual_loop, daemon=True, name='ManualLoop').start()
    print('  agent.main() running')

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
    signal.signal(signal.SIGINT,  _shutdown)

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
        shutdown_cleanup(wheels, camera, stop_event)


_SLIDERS_JSON = json.dumps(_CONFIG_SLIDERS)
_HTML = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Convoy Leader — Real</title>
<style>
 *{{margin:0;padding:0;box-sizing:border-box}}
 :root{{--bg:#13161a;--surface:#1a1d23;--border:#30363d;--text:#e6edf3;
       --muted:#8b949e;--accent:#1f6feb;--danger:#7d2622}}
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
 .mode-row{{display:flex;gap:6px}} .mode-row button{{flex:1}}
 .arrow-pad{{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;margin-top:8px}}
 .arrow-pad button{{height:40px;font-size:16px;padding:0}}
 .arrow-pad .sp{{visibility:hidden}}
 .sg{{margin-bottom:14px}}.sg:last-child{{margin-bottom:0}}
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
  <h1>Convoy Leader — Real Hardware</h1>
  <span class="sub">Lane Follow · Red Line Stop · Turns</span>
</div>
<div class="main">
  <div class="video-wrap">
    <img id="feed" src="/video">
    <button id="viewBtn" onclick="toggleView()" style="position:absolute;top:8px;right:8px;
      padding:4px 10px;font-size:11px;opacity:.75">Raw</button>
  </div>
  <div class="sidebar" id="sidebar">
    <div class="card">
      <div class="card-h">Status <span id="dot" style="width:7px;height:7px;border-radius:50%;
          background:var(--accent);display:inline-block"></span></div>
      <div id="status"></div>
    </div>
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
      <div class="mode-row" style="margin-top:8px">
        <button id="startBtn" onclick="setPaused(false)" style="flex:1;background:#2ea043;border-color:#2ea043">Start</button>
        <button id="stopBtn" onclick="setPaused(true)" style="flex:1">Stop</button>
      </div>
      <p style="color:var(--muted);font-size:11px;margin:6px 0 0">Arrow keys / WASD in manual mode.</p>
    </div>
    <div class="card"><div class="card-h">Red Line Config</div><div id="sliders-red_line"></div></div>
    <div class="card"><div class="card-h">Turn Config</div><div id="sliders-turn"></div></div>
    <div class="card"><div class="card-h">LED Config</div><div id="sliders-led"></div></div>
  </div>
</div>
<script>
const SLIDERS={_SLIDERS_JSON};
let keys={{up:false,down:false,left:false,right:false}};
let _rawView=false;
function post(u,b){{return fetch(u,{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(b||{{}})}});}}
function sendKeys(){{post('/keys',keys);}}
function press(k,on){{keys[k]=!!on;sendKeys();}}
function toggleView(){{_rawView=!_rawView;document.getElementById('feed').src=_rawView?'/raw':'/video';document.getElementById('viewBtn').className=_rawView?'on':'';document.getElementById('viewBtn').textContent=_rawView?'Overlay':'Raw';}}
function setMode(m){{post('/set_mode',{{mode:m}}).then(()=>{{
  document.getElementById('autoBtn').className=m==='auto'?'on':'';
  document.getElementById('manBtn').className=m==='manual'?'on':'';}});}}
function setPaused(p){{post(p?'/stop':'/start',{{}}).then(()=>{{
  document.getElementById('startBtn').className=p?'':'on';
  document.getElementById('stopBtn').className=p?'on':'';}});}}
const KMAP={{ArrowUp:'up',ArrowDown:'down',ArrowLeft:'left',ArrowRight:'right',w:'up',s:'down',a:'left',d:'right'}};
addEventListener('keydown',e=>{{if(KMAP[e.key]&&!keys[KMAP[e.key]]){{keys[KMAP[e.key]]=true;sendKeys();e.preventDefault();}}}});
addEventListener('keyup',e=>{{if(KMAP[e.key]){{keys[KMAP[e.key]]=false;sendKeys();e.preventDefault();}}}});

function buildSliders(){{
  const secs={{}};
  SLIDERS.forEach(s=>{{const sec=s[0];if(!secs[sec])secs[sec]=[];secs[sec].push(s);}});
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
    const paused=d.paused;
    dot.style.background=paused?'#e74c3c':(d.mode==='manual'?'var(--accent)':'#2ecc71');
    document.getElementById('startBtn').className=paused?'':'on';
    document.getElementById('stopBtn').className=paused?'on':'';
    document.getElementById('status').innerHTML=Object.entries(d).map(
      ([k,v])=>'<div class="row"><span class="k">'+k+'</span><span class="v">'+JSON.stringify(v)+'</span></div>'
    ).join('');
  }}).catch(()=>{{}});
}},400);
</script></body></html>"""


if __name__ == '__main__':
    sys.exit(main())
