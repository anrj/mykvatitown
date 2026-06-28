extends PathFollow3D

# Convoy leader (the NPC the follower chases).
#
# Drives smoothly around the loop and stops at the 4 intersection stop points
# (hardcoded from the Curve3D bake length). Approaching an intersection, the
# leader starts blinking its back LEDs amber at 2 Hz — about 3 seconds before
# it actually stops (within blink_radius metres of the stop point). The blink
# continues through the full STOP hold so the follower has plenty of time to
# confirm the turn direction, then runs its own open-loop arc to mimic the
# turn.
#
# The L/R LED mapping is hard-swapped so the side the FOLLOWER CAMERA sees
# matches the turn direction label ("R" turn → amber on the follower's right).
# If running this still looks inverted from your camera viewpoint, swap the
# two _set_led lines in _update_blink.

@export var cruise_speed: float = 0.22         # path units (metres) / second
@export var accel: float = 0.25                 # speed change / second (smoothness)
@export var stop_hold_s: float = 2.08           # pause duration at a stop point (matches leader_config_sim)
@export var stop_radius: float = 0.2            # how near a stop point counts as "arrived"
@export var blink_hz: float = 2.0               # LED blink cadence (slow enough for 24 fps sampling)
@export var blink_radius: float = 0.7           # metres before a stop point at which blink begins

# Hardcoded stop points (arclength along Curve3D_2rgth) at the 4 intersection
# corners. Tunable: adjust if the path is changed. Order follows lap direction.
@export var stop_points: PackedFloat32Array = PackedFloat32Array([4.5, 12.02, 14.08, 21.34])

var _speed: float = 0.0
var _started: bool = false                     # set by start() — dashboard button
var _stopped: bool = false                     # external stop (dashboard)
var _hold_until: float = 0.0
var _armed: bool = true                        # re-armed after leaving a stop point
var _blink_dir: String = "none"                # "L", "R", or "none" while signalling
var _blink_armed: bool = true                  # re-armed after leaving a blink zone
var _path_curve: Curve3D = null                 # cached parent Path3D curve (set in _ready)

# Node references to the back LED meshes (set in _ready, may be null if absent).
@onready var _back_led_l = get_node_or_null("DuckieBot_NPC/BackLED_L")
@onready var _back_led_r = get_node_or_null("DuckieBot_NPC/BackLED_R")


func _ready() -> void:
	rotation_mode = PathFollow3D.ROTATION_Y
	loop = true
	add_to_group("convoy_leader")              # so WheelCommandServer can find us
	# Cache the parent Path3D's curve (PathFollow3D has no `curve` member in Godot 4).
	var p = get_parent()
	if p is Path3D:
		_path_curve = p.curve
	# Defensive: if the scene names the LEDs differently, also look them up under the
	# follower's own NPC subtree naming convention.
	if _back_led_l == null:
		_back_led_l = get_node_or_null("BackLED_L")
	if _back_led_r == null:
		_back_led_r = get_node_or_null("BackLED_R")
	_leds_off()


func start() -> void:
	# Starts the leader, or resumes it after a dashboard stop.
	_started = true
	_stopped = false
	print("[ConvoyLeader] started")


func request_stop() -> void:
	# External stop (dashboard button) — eases to 0 and STAYS stopped
	# until start() is pressed again. (Stop-sign holds still use _hold_until.)
	_stopped = true
	print("[ConvoyLeader] external stop requested")


func _process(delta: float) -> void:
	var now := Time.get_unix_time_from_system()
	var near_stop := _near_stop_point()
	var near_blink := _near_blink_point()

	if not near_stop:
		_armed = true
	if not near_blink:
		_blink_armed = true

	# --- Blink onset: start signalling well before the actual stop ---
	# Sample turn direction the first time we enter the blink zone, then keep
	# the same _blink_dir through STOP until we leave the blink zone (a bit
	# past the intersection).
	if near_blink and _blink_armed and _started and not _stopped and _blink_dir == "none":
		_blink_dir = _compute_turn_dir()
		_blink_armed = false
		if _blink_dir != "none":
			print("[ConvoyLeader] BLINK ON  s=", progress, " dir=", _blink_dir)

	# --- Speed target ---
	var target := cruise_speed
	if not _started:
		target = 0.0                           # waiting for the "Start Leader" button
	elif _stopped:
		target = 0.0                           # dashboard stop — held until start()
	elif now < _hold_until:
		target = 0.0                           # currently holding at a stop sign
	elif near_stop and _armed:
		# Just arrived at a stop point — begin hold.
		_hold_until = now + stop_hold_s
		_armed = false
		target = 0.0
		print("[ConvoyLeader] STOP at s=", progress)

	# --- Blink offset: turn blink off only when we've left the blink zone ---
	if not near_blink and now >= _hold_until and _blink_dir != "none":
		print("[ConvoyLeader] BLINK OFF s=", progress, " dir=", _blink_dir)
		_blink_dir = "none"
		_leds_off()

	# smooth accel/decel toward target (no abrupt stops)
	if target > _speed:
		_speed = min(target, _speed + accel * delta)
	else:
		_speed = max(target, _speed - accel * delta)

	progress += _speed * delta

	# Drive LED blink while signalling.
	if _blink_dir != "none" and _back_led_l != null and _back_led_r != null:
		_update_blink(now)


func _near_stop_point() -> bool:
	for p in stop_points:
		if abs(progress - p) < stop_radius:
			return true
	return false


func _near_blink_point() -> bool:
	for p in stop_points:
		if abs(progress - p) < blink_radius:
			return true
	return false


# --- LED blink helpers -------------------------------------------------------

func _update_blink(now: float) -> void:
	var period: float = 1.0 / blink_hz if blink_hz > 0.0 else 0.5
	var on: bool = fmod(now, period) < (period * 0.5)
	# L/R mapping is hard-swapped so the side the FOLLOWER CAMERA sees matches
	# the turn label. (The leader's body mesh is rotated 180°, so its local
	# left/right is the camera's opposite.)
	if _blink_dir == "L":
		_set_led(_back_led_r, on)       # swapped: leader-left turn → blink leader's right LED
		_set_led(_back_led_l, false)
	elif _blink_dir == "R":
		_set_led(_back_led_l, on)       # swapped: leader-right turn → blink leader's left LED
		_set_led(_back_led_r, false)
	else:
		_leds_off()


func _set_led(node: Node, on: bool) -> void:
	if node == null:
		return
	# Toggle the mesh's visibility — the unshaded amber albedo renders when
	# visible, the LED vanishes when hidden. (emission_energy_multiplier has
	# no effect under SHADING_MODE_UNSHADED, so we use .visible instead.)
	node.visible = on


func _leds_off() -> void:
	_set_led(_back_led_l, false)
	_set_led(_back_led_r, false)


# --- Turn direction from path curvature --------------------------------------

func _compute_turn_dir() -> String:
	if _path_curve == null:
		return "none"
	# Sample path heading ~0.3m before and ~0.3m after the current progress.
	var total_len: float = _path_curve.get_baked_length()
	var s_before: float = max(0.0, progress - 0.3)
	var s_after: float = min(total_len - 0.001, progress + 0.3)
	var n_before: Vector3 = _path_curve.sample_baked(s_before)
	var n_after: Vector3 = _path_curve.sample_baked(s_after)
	if n_before.distance_to(n_after) < 1e-4:
		return "none"
	var h_before: float = atan2(n_after.x - n_before.x, n_after.z - n_before.z)
	var p2_after: Vector3 = _path_curve.sample_baked(min(total_len - 0.001, progress + 0.6))
	var h_after: float = atan2(p2_after.x - n_after.x, p2_after.z - n_after.z)
	var delta: float = (h_after - h_before + PI)
	delta = fmod(delta + PI, 2.0 * PI) - PI
	var tol: float = deg_to_rad(30.0)
	if delta > tol:
		return "L"
	elif delta < -tol:
		return "R"
	return "none"