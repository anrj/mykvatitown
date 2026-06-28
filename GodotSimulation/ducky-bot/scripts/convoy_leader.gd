extends PathFollow3D

# Convoy leader (the NPC the follower chases).
#
# Drives smoothly around the loop and stops at the 4 intersection stop points
# (hardcoded from the Curve3D bake length). While stopped at an intersection,
# the leader blinks its back LEDs amber at 5 Hz to signal the turn direction
# (L = left LED, R = right LED). The turn direction is sampled from the path
# curvature just ahead of the stop point, so the leader always blinks the
# correct side matching the upcoming corner.
#
# The PathFollow3D itself drives the leader along Curve3D — there is no
# open-loop arc here. The follower sees the amber blink, then runs its own
# open-loop arc (turn_*_pwm / turn_*_s) to mimic the leader's turn.

@export var cruise_speed: float = 0.22         # path units (metres) / second
@export var accel: float = 0.25                 # speed change / second (smoothness)
@export var stop_hold_s: float = 2.08           # pause duration at a stop point (matches leader_config_sim)
@export var stop_radius: float = 0.2            # how near a stop point counts as "arrived"
@export var blink_hz: float = 5.0               # LED blink cadence during STOP

# Hardcoded stop points (arclength along Curve3D_2rgth) at the 4 intersection
# corners. Tunable: adjust if the path is changed. Order follows lap direction.
@export var stop_points: PackedFloat32Array = PackedFloat32Array(4.5, 12.02, 14.08, 21.34)

var _speed: float = 0.0
var _started: bool = false                     # set by start() — dashboard button
var _stopped: bool = false                     # external stop (dashboard)
var _hold_until: float = 0.0
var _armed: bool = true                        # re-armed after leaving a stop point
var _blink_dir: String = "none"                # "L", "R", or "none" while holding

# Node references to the back LED meshes (set in _ready, may be null if absent).
@onready var _back_led_l = get_node_or_null("DuckieBot_NPC/BackLED_L")
@onready var _back_led_r = get_node_or_null("DuckieBot_NPC/BackLED_R")


func _ready() -> void:
	rotation_mode = PathFollow3D.ROTATION_Y
	loop = true
	add_to_group("convoy_leader")              # so WheelCommandServer can find us
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
	var near := _near_stop_point()
	if not near:
		_armed = true                          # left the sign, ready for next lap

	var target := cruise_speed
	if not _started:
		target = 0.0                           # waiting for the "Start Leader" button
	elif _stopped:
		target = 0.0                           # dashboard stop — held until start()
	elif now < _hold_until:
		target = 0.0                           # currently holding at a stop sign
	elif near and _armed:
		# Just arrived at a stop point — begin hold.
		_hold_until = now + stop_hold_s
		_armed = false
		target = 0.0
		_blink_dir = _compute_turn_dir()
		print("[ConvoyLeader] STOP at s=", progress, " blink=", _blink_dir)

	if now >= _hold_until:
		# Past the hold window — stop blinking.
		if _blink_dir != "none":
			_blink_dir = "none"
			_leds_off()

	# smooth accel/decel toward target (no abrupt stops)
	if target > _speed:
		_speed = min(target, _speed + accel * delta)
	else:
		_speed = max(target, _speed - accel * delta)

	progress += _speed * delta

	# Drive LED blink while holding.
	if _blink_dir != "none" and _back_led_l != null and _back_led_r != null:
		_update_blink(now)


func _near_stop_point() -> bool:
	for p in stop_points:
		if abs(progress - p) < stop_radius:
			return true
	return false


# --- LED blink helpers -------------------------------------------------------

func _update_blink(now: float) -> void:
	var period: float = 1.0 / blink_hz if blink_hz > 0.0 else 0.2
	var on: bool = fmod(now, period) < (period * 0.5)
	if _blink_dir == "L":
		_set_led(_back_led_l, on)
		_set_led(_back_led_r, false)
	elif _blink_dir == "R":
		_set_led(_back_led_r, on)
		_set_led(_back_led_l, false)
	else:
		_leds_off()


func _set_led(node: Node, on: bool) -> void:
	if node == null:
		return
	# StandardMaterial3D emissive toggle. Assumes the mesh uses a material override
	# with emission_energy_multiplier; 1.0 when on, 0.0 when off.
	var mat: StandardMaterial3D = node.get_surface_override_material(0) if node is MeshInstance3D else null
	if mat == null and node is MeshInstance3D:
		# Lazy-create a per-instance amber emissive material.
		mat = StandardMaterial3D.new()
		mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
		mat.cull_mode = BaseMaterial3D.CULL_MODE_DISABLED
		mat.albedo_color = Color(1.0, 0.6, 0.0, 1.0)
		mat.emission_enabled = true
		mat.emission = Color(1.0, 0.6, 0.0, 1.0)
		mat.emission_energy_multiplier = 1.0 if on else 0.0
		(node as MeshInstance3D).set_surface_override_material(0, mat)
		return
	if mat != null:
		mat.emission_energy_multiplier = 1.0 if on else 0.0


func _leds_off() -> void:
	_set_led(_back_led_l, false)
	_set_led(_back_led_r, false)


# --- Turn direction from path curvature --------------------------------------

func _compute_turn_dir() -> String:
	# Sample path heading ~0.3m before and ~0.3m after the current progress.
	var total_len: float = curve.get_baked_length()
	var s_before: float = max(0.0, progress - 0.3)
	var s_after: float = min(total_len - 0.001, progress + 0.3)
	var n_before: Vector3 = curve.sample_baked(s_before)
	var n_after: Vector3 = curve.sample_baked(s_after)
	if n_before.distance_to(n_after) < 1e-4:
		return "none"
	var h_before: float = atan2(n_after.x - n_before.x, n_after.z - n_before.z)
	var p2_after: Vector3 = curve.sample_baked(min(total_len - 0.001, progress + 0.6))
	var h_after: float = atan2(p2_after.x - n_after.x, p2_after.z - n_after.z)
	var delta: float = (h_after - h_before + PI)
	delta = fmod(delta + PI, 2.0 * PI) - PI
	var tol: float = deg_to_rad(30.0)
	if delta > tol:
		return "L"
	elif delta < -tol:
		return "R"
	return "none"