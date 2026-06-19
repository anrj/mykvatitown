extends PathFollow3D

# Convoy leader (the NPC the follower chases).
#
# It drives smoothly around the loop. There are NO timed phases: it cruises
# at a constant speed and only eases to a stop at "stop points" — positions
# along the path where a stop sign stands — then eases back up. Stops are
# smooth (ramped accel/decel), never abrupt. The circle-grid plate on the
# leader's back is a separate scene node (DotPlate / convoy_dotplate.gd).

@export var cruise_speed: float = 0.22         # path units (metres) / second
@export var accel: float = 0.25                # speed change / second (smoothness)
@export var stop_hold_s: float = 2.0           # pause duration at a stop point
@export var stop_radius: float = 0.35          # how near a stop point counts as "arrived"

# Progress distances (metres along the curve) where stop signs stand.
# Empty for Milestone 1 (no signs yet) => constant cruise around the loop.
# In Milestone 2, set these to match the signs you place in the scene.
@export var stop_points: PackedFloat32Array = PackedFloat32Array()

var _speed: float = 0.0
var _started: bool = false                     # set to true by start() — called from the dashboard button
var _stopped: bool = false                     # external stop (dashboard) — stays until start() clears it
var _hold_until: float = 0.0
var _armed: bool = true                        # re-armed after leaving a stop point


func _ready() -> void:
	rotation_mode = PathFollow3D.ROTATION_Y
	loop = true
	add_to_group("convoy_leader")              # so WheelCommandServer can find us


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
		_hold_until = now + stop_hold_s        # just arrived -> begin hold
		_armed = false
		target = 0.0

	# smooth accel/decel toward target (no abrupt stops)
	if target > _speed:
		_speed = min(target, _speed + accel * delta)
	else:
		_speed = max(target, _speed - accel * delta)

	progress += _speed * delta


func _near_stop_point() -> bool:
	for p in stop_points:
		if abs(progress - p) < stop_radius:
			return true
	return false
