extends Control

# 暴露变量，方便你在右侧面板直接调颜色和粗细
@export var radius: float = 40.0
@export var thickness: float = 8.0
@export var bg_color: Color = Color(0.2, 0.2, 0.2) # 默认全灰
@export var fill_color: Color = Color(0.2, 0.7, 1.0) # 浅蓝色

# 进度值 0.0 到 1.0
var progress: float = 0.0:
	set(value):
		progress = clamp(value, 0.0, 1.0)
		queue_redraw() # 每次进度改变，通知引擎重新执行 _draw() 绘制

func _draw():
	var center = size / 2.0
	var start_angle = -PI / 2.0 # 从正上方 (12点钟方向) 开始
	
	# 1. 画底部的灰色圆环 (画满一整圈 360度 = TAU)
	draw_arc(center, radius, 0, TAU, 64, bg_color, thickness, true)
	
	# 2. 画蓝色的进度圆环
	if progress > 0.0:
		var end_angle = start_angle + (progress * TAU)
		draw_arc(center, radius, start_angle, end_angle, 64, fill_color, thickness, true)
