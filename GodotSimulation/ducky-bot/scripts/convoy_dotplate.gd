@tool
class_name ConvoyDotplate
extends MeshInstance3D

# The Duckietown circle-grid plate (cols x rows black dots on white) on the
# leader's back. It builds its own quad + dot texture procedurally, so it
# renders live in the Godot editor (drag it onto the NPC's back to position
# it) AND at runtime. Two-sided + unshaded so the follower's
# cv2.findCirclesGrid detects it regardless of facing.

@export var grid_cols: int = 7:
	set(v): grid_cols = v; _build()
@export var grid_rows: int = 3:
	set(v): grid_rows = v; _build()
@export var plate_width_m: float = 0.12:
	set(v): plate_width_m = v; _build()


func _ready() -> void:
	_build()


func _build() -> void:
	if grid_cols <= 0 or grid_rows <= 0:
		return
	var height_m: float = plate_width_m * float(grid_rows) / float(grid_cols)
	var quad := QuadMesh.new()
	quad.size = Vector2(plate_width_m, height_m)
	mesh = quad

	var mat := StandardMaterial3D.new()
	mat.albedo_texture = _make_texture(grid_cols, grid_rows)
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	mat.cull_mode = BaseMaterial3D.CULL_DISABLED          # visible from both sides
	mat.texture_filter = BaseMaterial3D.TEXTURE_FILTER_NEAREST
	material_override = mat


func _make_texture(cols: int, rows: int) -> ImageTexture:
	var margin := 16
	var cell := 48
	var w := margin * 2 + cell * cols
	var h := margin * 2 + cell * rows
	var img := Image.create(w, h, false, Image.FORMAT_RGB8)
	img.fill(Color.WHITE)

	var r := int(cell * 0.32)
	for cx in range(cols):
		for cy in range(rows):
			_disc(img, margin + cell / 2 + cx * cell, margin + cell / 2 + cy * cell, r)
	return ImageTexture.create_from_image(img)


func _disc(img: Image, cx: int, cy: int, r: int) -> void:
	for y in range(cy - r, cy + r + 1):
		for x in range(cx - r, cx + r + 1):
			if x < 0 or y < 0 or x >= img.get_width() or y >= img.get_height():
				continue
			var dx := x - cx
			var dy := y - cy
			if dx * dx + dy * dy <= r * r:
				img.set_pixel(x, y, Color.BLACK)
