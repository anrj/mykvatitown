from .base import render_template, BASE_CSS, BASE_JS

_EXTRA_CSS = '''
.distance-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.distance-badge.LOST      { background: rgba(248,81,73,0.15);  border: 1px solid var(--accent-red);    color: var(--accent-red); }
.distance-badge.TOO_CLOSE { background: rgba(248,81,73,0.15);  border: 1px solid var(--accent-red);    color: var(--accent-red); }
.distance-badge.CLOSE     { background: rgba(210,153,34,0.15); border: 1px solid var(--accent-orange); color: var(--accent-orange); }
.distance-badge.GOOD      { background: rgba(63,185,80,0.15);  border: 1px solid var(--accent-green);  color: var(--accent-green); }
.distance-badge.FAR       { background: rgba(31,111,235,0.15); border: 1px solid var(--accent-blue);   color: var(--accent-blue); }

.speed-bar-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
    font-size: 12px;
}
.speed-bar-label { width: 38px; color: var(--text-muted); flex-shrink: 0; }
.speed-bar-track {
    flex: 1;
    height: 8px;
    background: var(--bg-sidebar);
    border-radius: 4px;
    overflow: hidden;
    border: 1px solid var(--border-color);
}
.speed-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: var(--accent-blue);
    transition: width 0.15s;
}
.speed-bar-value { width: 38px; text-align: right; color: var(--text-secondary); flex-shrink: 0; font-variant-numeric: tabular-nums; }

.model-status { padding: 6px 10px; border-radius: 4px; font-size: 12px; margin-bottom: 10px; }
.model-status.ok  { background: rgba(63,185,80,0.1);  border: 1px solid rgba(63,185,80,0.3);  color: var(--accent-green); }
.model-status.err { background: rgba(248,81,73,0.1);  border: 1px solid rgba(248,81,73,0.3);  color: var(--accent-red); }
.model-status.loading { background: rgba(210,153,34,0.1); border: 1px solid rgba(210,153,34,0.3); color: #d6a63a; }

.reason-text { font-size: 11px; color: var(--text-muted); margin-top: 4px; font-style: italic; }
'''

_CONTENT = '''
    <div class="container">
        <div class="video-section">
            <img src="/video" class="stream">
        </div>

        <div class="controls-section">

            <!-- Drive Control -->
            <div class="card">
                <div class="card-header">Drive Control</div>
                <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px">
                    <span id="run-indicator" style="width:14px;height:14px;border-radius:50%;background:#e74c3c;flex-shrink:0;display:inline-block"></span>
                    <span id="run-label" style="font-size:14px;font-weight:600;color:var(--text-secondary)">STOPPED</span>
                </div>
                <button class="button success" onclick="post('/start')">▶ Start</button>
                <button class="button danger"  onclick="post('/stop')">■ Stop</button>
                <button class="button"         onclick="post('/reset')">↺ Reset Tracker</button>
                <div id="ctrl-status" class="status"></div>
            </div>

            <!-- Model Status -->
            <div class="card">
                <div class="card-header">System</div>
                <div id="model-status" class="model-status loading">Loading detection model...</div>
                <div class="config-item">
                    <span class="config-label">Host</span>
                    <span class="config-value">{{ hostname }}</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Lane frames</span>
                    <span class="config-value" id="lane-frames">0</span>
                </div>
            </div>

            <!-- Target Status -->
            <div class="card">
                <div class="card-header">Target Truck</div>
                <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px">
                    <span style="font-size:13px;color:var(--text-secondary)">Distance state</span>
                    <span id="distance-badge" class="distance-badge LOST">LOST</span>
                </div>
                <div class="stats-grid" style="grid-template-columns:1fr 1fr 1fr;">
                    <div class="stat-box">
                        <div class="stat-value" id="target-score">—</div>
                        <div class="stat-label">Score</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="target-area">—</div>
                        <div class="stat-label">Area px²</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="target-bottom">—</div>
                        <div class="stat-label">Bottom Y</div>
                    </div>
                </div>
                <div class="reason-text" id="target-reason">waiting for detections...</div>
            </div>

            <!-- Speeds -->
            <div class="card">
                <div class="card-header">
                    Wheel Speeds
                    <span id="multiplier-badge" style="font-size:11px;font-weight:400;color:var(--text-muted)">×1.00</span>
                </div>
                <div class="speed-bar-row">
                    <span class="speed-bar-label">Lane L</span>
                    <div class="speed-bar-track"><div class="speed-bar-fill" id="bar-lane-l" style="width:0%"></div></div>
                    <span class="speed-bar-value" id="val-lane-l">0.000</span>
                </div>
                <div class="speed-bar-row">
                    <span class="speed-bar-label">Lane R</span>
                    <div class="speed-bar-track"><div class="speed-bar-fill" id="bar-lane-r" style="width:0%"></div></div>
                    <span class="speed-bar-value" id="val-lane-r">0.000</span>
                </div>
                <div class="speed-bar-row" style="margin-top:6px">
                    <span class="speed-bar-label">Final L</span>
                    <div class="speed-bar-track"><div class="speed-bar-fill" id="bar-final-l" style="width:0%;background:var(--accent-green)"></div></div>
                    <span class="speed-bar-value" id="val-final-l">0.000</span>
                </div>
                <div class="speed-bar-row">
                    <span class="speed-bar-label">Final R</span>
                    <div class="speed-bar-track"><div class="speed-bar-fill" id="bar-final-r" style="width:0%;background:var(--accent-green)"></div></div>
                    <span class="speed-bar-value" id="val-final-r">0.000</span>
                </div>
                <div class="reason-text" id="command-reason">—</div>
            </div>

            <!-- Tuning -->
            <div class="card">
                <div class="card-header">Speed Multipliers</div>
                <div class="slider-group">
                    <div class="slider-label">
                        <span>Close (slow down)</span>
                        <span id="close-val">0.40</span>
                    </div>
                    <div class="slider-controls">
                        <input type="range" class="slider" id="close-slider" min="0" max="1" step="0.05" value="0.40"
                               oninput="document.getElementById('close-val').textContent=parseFloat(this.value).toFixed(2); sendConfig()">
                    </div>
                </div>
                <div class="slider-group">
                    <div class="slider-label">
                        <span>Good (maintain)</span>
                        <span id="good-val">1.00</span>
                    </div>
                    <div class="slider-controls">
                        <input type="range" class="slider" id="good-slider" min="0" max="1.5" step="0.05" value="1.00"
                               oninput="document.getElementById('good-val').textContent=parseFloat(this.value).toFixed(2); sendConfig()">
                    </div>
                </div>
                <div class="slider-group">
                    <div class="slider-label">
                        <span>Far (catch up)</span>
                        <span id="far-val">1.20</span>
                    </div>
                    <div class="slider-controls">
                        <input type="range" class="slider" id="far-slider" min="0" max="2" step="0.05" value="1.20"
                               oninput="document.getElementById('far-val').textContent=parseFloat(this.value).toFixed(2); sendConfig()">
                    </div>
                </div>
                <div id="tune-status" class="status"></div>
            </div>

        </div>
    </div>
'''

_EXTRA_JS = '''
function post(path) {
    fetch(path, {method: 'POST'})
        .then(r => r.json())
        .then(data => {
            const msg = data.status || JSON.stringify(data);
            showStatus('ctrl-status', msg, 'success');
        })
        .catch(() => showStatus('ctrl-status', 'Request failed', 'error'));
}

function setBar(barId, valId, value, maxVal) {
    const pct = Math.min(100, Math.max(0, (value / maxVal) * 100));
    document.getElementById(barId).style.width = pct + '%';
    document.getElementById(valId).textContent = value.toFixed(3);
}

function updateStatus() {
    fetch('/status')
        .then(r => r.json())
        .then(data => {
            // Running indicator
            const running = data.running;
            document.getElementById('run-indicator').style.background = running ? '#3fb950' : '#e74c3c';
            document.getElementById('run-label').textContent = running ? 'RUNNING' : 'STOPPED';

            // Model
            const modelEl = document.getElementById('model-status');
            if (data.model_loaded) {
                modelEl.className = 'model-status ok';
                modelEl.textContent = '✓ Detection model loaded';
            } else if (data.model_load_error) {
                modelEl.className = 'model-status err';
                modelEl.textContent = '✗ ' + data.model_load_error;
            } else {
                modelEl.className = 'model-status loading';
                modelEl.textContent = 'Loading detection model...';
            }

            document.getElementById('lane-frames').textContent = data.lane_frame_count || 0;

            // Target
            const t = data.target;
            const state = t ? t.distance_state : 'LOST';
            const badge = document.getElementById('distance-badge');
            badge.textContent = state;
            badge.className = 'distance-badge ' + state;

            document.getElementById('target-score').textContent  = t && t.found ? t.score.toFixed(2) : '—';
            document.getElementById('target-area').textContent   = t && t.found ? t.area : '—';
            document.getElementById('target-bottom').textContent = t && t.found ? t.bottom_y : '—';
            document.getElementById('target-reason').textContent = t ? t.reason : '—';

            // Speeds
            const laneMax = 1.5;
            setBar('bar-lane-l',  'val-lane-l',  data.lane_left  || 0, laneMax);
            setBar('bar-lane-r',  'val-lane-r',  data.lane_right || 0, laneMax);

            const cmd = data.command;
            const mult = cmd ? cmd.speed_multiplier : 0;
            document.getElementById('multiplier-badge').textContent = '×' + mult.toFixed(2);
            document.getElementById('command-reason').textContent = cmd ? cmd.reason : '—';

            setBar('bar-final-l', 'val-final-l', cmd ? cmd.left_speed  : 0, laneMax);
            setBar('bar-final-r', 'val-final-r', cmd ? cmd.right_speed : 0, laneMax);
        })
        .catch(() => {});
}

function sendConfig() {
    const data = {
        close_multiplier: parseFloat(document.getElementById('close-slider').value),
        good_multiplier:  parseFloat(document.getElementById('good-slider').value),
        far_multiplier:   parseFloat(document.getElementById('far-slider').value),
    };
    postJSON('/update_config', data)
        .then(() => showStatus('tune-status', 'Config updated', 'success'))
        .catch(() => showStatus('tune-status', 'Update failed', 'error'));
}

setInterval(updateStatus, 300);
updateStatus();
'''


CONVOYING_TEMPLATE = render_template(
    title    = "Convoying — Truck Following",
    subtitle = "Lane servoing handles steering · Convoying adjusts speed based on truck distance",
    content_html = _CONTENT,
    extra_css    = _EXTRA_CSS,
    extra_js     = _EXTRA_JS,
)