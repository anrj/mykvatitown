"""Shared keyboard manual drive for Godot simulation servers."""

import threading
import time
from typing import Callable

from tasks.introduction.packages.manual_drive import get_motor_speeds


class ManualDriveController:
    """Arrow-key / WASD manual control (same as the introduction task)."""

    def __init__(self):
        self.manual_mode = False
        self._keys = {'up': False, 'down': False, 'left': False, 'right': False}
        self._lock = threading.Lock()
        self._keys_stamp = 0.0
        self.last_left = 0.0
        self.last_right = 0.0

    def set_mode(self, mode: str) -> str:
        self.manual_mode = (mode == 'manual')
        return 'manual' if self.manual_mode else 'auto'

    def update_keys(self, data: dict) -> None:
        with self._lock:
            for k in self._keys:
                self._keys[k] = bool(data.get(k, False))
        self._keys_stamp = time.time()

    def snapshot_keys(self) -> dict:
        with self._lock:
            return dict(self._keys)

    def run_loop(self, stop_event, set_wheels: Callable[[float, float], None],
                 poll_s: float = 0.05, key_timeout_s: float = 0.5) -> None:
        while not stop_event.is_set():
            if not self.manual_mode:
                time.sleep(poll_s)
                continue
            if time.time() - self._keys_stamp > key_timeout_s:
                with self._lock:
                    for k in self._keys:
                        self._keys[k] = False
            keys = self.snapshot_keys()
            left, right = get_motor_speeds(keys)
            self.last_left = float(max(-1.0, min(1.0, left)))
            self.last_right = float(max(-1.0, min(1.0, right)))
            set_wheels(self.last_left, self.last_right)
            time.sleep(poll_s)


MANUAL_KEY_JS = '''
let keys={up:false,down:false,left:false,right:false};
function sendKeys(){post('/keys',keys);}
function press(k,on){keys[k]=!!on;sendKeys();}
const KMAP={ArrowUp:'up',ArrowDown:'down',ArrowLeft:'left',ArrowRight:'right',
            w:'up',s:'down',a:'left',d:'right',W:'up',S:'down',A:'left',D:'right'};
addEventListener('keydown',e=>{if(KMAP[e.key]&&!keys[KMAP[e.key]]){keys[KMAP[e.key]]=true;sendKeys();e.preventDefault();}});
addEventListener('keyup',e=>{if(KMAP[e.key]){keys[KMAP[e.key]]=false;sendKeys();e.preventDefault();}});
'''

MANUAL_PAD_HTML = '''
      <div class="arrow-pad">
        <div class="sp"></div><button onmousedown="press('up',1)" onmouseup="press('up',0)" onmouseleave="press('up',0)">▲</button><div class="sp"></div>
        <button onmousedown="press('left',1)" onmouseup="press('left',0)" onmouseleave="press('left',0)">◀</button>
        <button onmousedown="press('down',1)" onmouseup="press('down',0)" onmouseleave="press('down',0)">▼</button>
        <button onmousedown="press('right',1)" onmouseup="press('right',0)" onmouseleave="press('right',0)">▶</button>
      </div>
      <p style="color:var(--muted);font-size:11px;margin:6px 0 0">Arrow keys / WASD in manual mode.</p>
'''

MANUAL_PAD_CSS = '''
 .arrow-pad{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;margin-top:8px}
 .arrow-pad button{height:40px;font-size:16px;padding:0}
 .arrow-pad .sp{visibility:hidden}
'''
