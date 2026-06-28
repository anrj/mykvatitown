extends PathFollow3D

# Convoy leader (the NPC the follower chases).
#
# Drives smoothly around the loop and stops at the 4 intersection stop points
# (hardcoded from the Curve3D bake length). The follower handles intersection
# navigation purely via lane servoing -- no LED blink signalling.

@export var cruise_speed: float = 0.22
@export var accel: float = 0.25
@export var stop_hold_s: float = 2.08
@export var stop_radius: float = 0.2

@export var stop_points: PackedFloat32Array = PackedFloat32Array([4.5, 12.02, 14.08, 21.34])

var _speed: float = 0.0
var _started: bool = false
var _stopped: bool = false
var _hold_until: float = 0.0
var _armed: bool = true


func _ready() -> void:
	rotation_mode = PathFollow3D.ROTATION_Y
	loop = true
	add_to_group("convoy_leader")


func start() -> void:
	_started = true
	_stopped = false
	print("[ConvoyLeader] started")


func request_stop() -> void:
	_stopped = true
	print("[ConvoyLeader] external stop requested")


func _process(delta: float) -> void:
	var now := Time.get_unix_time_from_system()
	var near_stop := _near_stop_point()

	if not near_stop:
		_armed = true

	var target := cruise_speed
	if not _started:
		target = 0.0
	elif _stopped:
		target = 0.0
	elif now < _hold_until:
		target = 0.0
	elif near_stop and _armed:
		_hold_until = now + stop_hold_s
		_armed = false
		target = 0.0
		print("[ConvoyLeader] STOP at s=", progress)

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
