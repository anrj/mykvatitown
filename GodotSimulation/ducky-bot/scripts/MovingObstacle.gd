extends Node3D

@export var speed: float = 0.16
@export var reach_distance: float = 0.05
@export var rotation_offset_degrees: float = 180.0

var _points: Array[Vector3] = []
var _current_index: int = 0
var _finished: bool = false


func _ready() -> void:
	# Lead truck starts close enough to follower.
	# It goes straight a bit longer, turns left smoothly,
	# then continues straight after the turn.

	_points = [
	# straight before crossroad
	Vector3(4.17, 0.08, 2.30),
	Vector3(4.17, 0.08, 2.45),
	Vector3(4.17, 0.08, 2.60),
	Vector3(4.17, 0.08, 2.72),

	# smooth left turn starts earlier here
	Vector3(4.22, 0.08, 2.84),
	Vector3(4.30, 0.08, 2.94),
	Vector3(4.42, 0.08, 3.02),
	Vector3(4.58, 0.08, 3.08),
	Vector3(4.76, 0.08, 3.12),
	Vector3(4.96, 0.08, 3.14),
	Vector3(5.15, 0.08, 3.15),

	# after turn: keep going straight in the new lane
	Vector3(5.35, 0.08, 3.15),
	Vector3(5.60, 0.08, 3.15),
	Vector3(5.85, 0.08, 3.15),
	Vector3(6.10, 0.08, 3.15),
	Vector3(6.35, 0.08, 3.15),
	Vector3(6.60, 0.08, 3.15),
]

	global_position = _points[0]
	_current_index = 1
	_finished = false

	_face_next_point()

	print("[LeadTruckDriver] Lead truck started")


func _physics_process(delta: float) -> void:
	if _finished:
		return

	if _current_index >= _points.size():
		_finished = true
		print("[LeadTruckDriver] Lead truck reached end and stopped")
		return

	var target := _points[_current_index]
	var to_target := target - global_position
	var distance := to_target.length()

	if distance <= reach_distance:
		_current_index += 1

		if _current_index >= _points.size():
			_finished = true
			print("[LeadTruckDriver] Lead truck reached end and stopped")
			return

		_face_next_point()
		return

	var direction := to_target.normalized()

	global_position += direction * speed * delta

	look_at(global_position + direction, Vector3.UP)
	rotate_y(deg_to_rad(rotation_offset_degrees))


func _face_next_point() -> void:
	if _current_index >= _points.size():
		return

	var direction := (_points[_current_index] - global_position).normalized()

	if direction.length() > 0.0:
		look_at(global_position + direction, Vector3.UP)
		rotate_y(deg_to_rad(rotation_offset_degrees))