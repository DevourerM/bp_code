@tool
extends GraphNode

const PORT_TYPE_TENSOR = 0
const PORT_TYPE_PARAM = 1
const COLOR_TENSOR = Color(0.3, 0.7, 1.0) 
const COLOR_PARAM = Color(0.5, 0.9, 0.5)  

signal parameter_changed
signal enter_subgraph_requested(node_name: String)

var config: Dictionary = {}
var ui_controls: Dictionary = {}
var _is_updating_auto: bool = false 

func setup_node(new_config: Dictionary):
	config = new_config.duplicate(true)
	if is_node_ready(): _rebuild_ui()

func _ready():
	_apply_premium_theme()
	if not config.is_empty(): _rebuild_ui()

func _apply_premium_theme():
	var bg = StyleBoxFlat.new()
	bg.bg_color = Color(0.12, 0.12, 0.14, 0.95)
	bg.corner_radius_bottom_left = 10; bg.corner_radius_bottom_right = 10
	bg.border_width_left = 2; bg.border_width_right = 2; bg.border_width_bottom = 2
	bg.border_color = Color(0.08, 0.08, 0.1)
	bg.content_margin_left = 16; bg.content_margin_right = 16
	bg.content_margin_bottom = 16; bg.content_margin_top = 8
	var bg_sel = bg.duplicate()
	bg_sel.border_color = Color(0.4, 0.6, 1.0) 
	
	var tb = StyleBoxFlat.new()
	tb.bg_color = Color(0.18, 0.20, 0.25, 0.95)
	tb.corner_radius_top_left = 10; tb.corner_radius_top_right = 10
	tb.content_margin_left = 16; tb.content_margin_right = 16 
	tb.content_margin_top = 10; tb.content_margin_bottom = 10
	var tb_sel = tb.duplicate()
	tb_sel.bg_color = Color(0.25, 0.35, 0.5, 0.95) 
	
	add_theme_stylebox_override("panel", bg)
	add_theme_stylebox_override("panel_selected", bg_sel)
	add_theme_stylebox_override("titlebar", tb)
	add_theme_stylebox_override("titlebar_selected", tb_sel)

func _rebuild_ui():
	ui_controls.clear()
	clear_all_slots()
	for child in get_children(): child.free() 
	custom_minimum_size = Vector2(220, 0) 
	reset_size()
	self.title = config.get("name", "Unknown Function")
	
	# === 动态生成引脚 ===
	var ins = config.get("inputs", []).duplicate()
	if config.name == "Group":
		ins.clear()
		var count = 1
		if config.has("params"):
			for p in config.params:
				if p.name == "input_count":
					count = int(p.get("value", p.get("default", 1)))
		for i in range(count):
			ins.append("input" + str(i))
	elif ins.is_empty() and config.get("main_in", "") != "":
		ins.append(config.get("main_in", ""))
		
	var out_str = config.get("main_out", "")
	var has_out = out_str != ""
	var max_io = max(ins.size(), 1 if has_out else 0)
	
	for i in range(max_io):
		var io_container = HBoxContainer.new()
		io_container.add_theme_constant_override("separation", 16)
		if i < ins.size():
			var in_label = Label.new(); in_label.text = ins[i]
			io_container.add_child(in_label)
		var spacer = Control.new(); spacer.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		io_container.add_child(spacer)
		if i == 0 and has_out:
			var out_label = Label.new(); out_label.text = out_str
			out_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
			io_container.add_child(out_label)
		add_child(io_container)
		set_slot(i, i < ins.size(), 0, Color.AQUA, i == 0 and has_out, 0, Color.AQUA)

	# === 生成参数 ===
	if config.has("params"):
		for p in config["params"]:
			var param_container = HBoxContainer.new()
			var p_label = Label.new(); p_label.text = p.name
			p_label.size_flags_horizontal = SIZE_EXPAND_FILL 
			param_container.add_child(p_label)
			
			var ctrl: Control
			if p.type == "enum":
				var ob = OptionButton.new()
				for opt in p.get("options", []): ob.add_item(str(opt))
				for i in range(ob.item_count):
					if ob.get_item_text(i) == str(p.get("value", p.get("default", ""))): ob.select(i); break
				ob.item_selected.connect(func(_idx): 
					p["value"] = ob.get_item_text(_idx)
					if not _is_updating_auto: parameter_changed.emit()
				)
				ctrl = ob
			elif p.type == "code":
				var btn = Button.new()
				btn.text = "📝 编辑代码..."
				btn.pressed.connect(func(): _open_code_editor(p))
				ctrl = btn
				
			elif p.type == "int" or p.type == "float":
				var sb = SpinBox.new(); sb.allow_greater = true
				if p.name == "input_count": sb.min_value = 1 
				sb.step = 1 if p.type == "int" else 0.01
				sb.value = float(p.get("value", p.get("default", 1 if p.name == "input_count" else 0)))
				
				sb.value_changed.connect(func(_v): 
					p["value"] = _v
					if p.name == "input_count": _rebuild_ui()
					if not _is_updating_auto: parameter_changed.emit()
				)
				ctrl = sb
			else: 
				var le = LineEdit.new()
				le.text = str(p.get("value", p.get("default", "")))
				le.expand_to_text_length = true
				le.text_changed.connect(func(_t): 
					p["value"] = _t
					if not _is_updating_auto: parameter_changed.emit()
				)
				ctrl = le
			
			ctrl.custom_minimum_size.x = 90
			param_container.add_child(ctrl)
			ui_controls[p.name] = ctrl
			add_child(param_container)
			set_slot(get_child_count()-1, true, 1, Color.LAWN_GREEN, false, 0, Color.WHITE)

	# === 子图按钮 ===
	if config.name in ["Group", "Loop"]:
		var btn = Button.new()
		btn.text = "进入子层级 [Enter]"
		btn.add_theme_color_override("font_color", Color(1, 0.8, 0.2))
		btn.pressed.connect(func(): enter_subgraph_requested.emit(self.name))
		add_child(btn)
		set_slot(get_child_count()-1, false, 0, Color.WHITE, false, 0, Color.WHITE)

func get_current_params() -> Dictionary:
	var current_params = {}
	if not config.has("params"): return current_params
	for p in config["params"]:
		var c = ui_controls.get(p.name)
		if c:
			var val = ""
			if c is SpinBox: val = str(c.value)
			elif c is LineEdit: val = c.text
			elif c is OptionButton: val = c.get_item_text(c.selected)
			current_params[p.name] = {"type": p.type, "value": val}
	return current_params

func apply_auto_params(patch):
	_is_updating_auto = true 
	for k in patch:
		if ui_controls.has(k):
			var c = ui_controls[k]
			if c is LineEdit: c.text = str(patch[k]); c.editable = false
			elif c is SpinBox: c.value = float(patch[k]); c.editable = false
			elif c is OptionButton:
				for i in range(c.item_count):
					if c.get_item_text(i) == str(patch[k]): c.select(i); break
				c.disabled = true
			c.modulate = Color.GREEN
	_is_updating_auto = false

func reset_auto_params():
	for c in ui_controls.values(): 
		if c is LineEdit or c is SpinBox: c.editable = true
		if c is OptionButton: c.disabled = false
		c.modulate = Color.WHITE

# ==========================================
# 弹出式多行代码编辑器逻辑
# ==========================================
func _open_code_editor(p: Dictionary):
	var win = Window.new()
	win.title = "编辑 Python 代码 - " + p.name
	win.size = Vector2(700, 500)
	win.exclusive = true # 模态窗口，必须关掉才能点别处
	
	var vbox = VBoxContainer.new()
	vbox.set_anchors_and_offsets_preset(PRESET_FULL_RECT)
	win.add_child(vbox)
	
	var text_edit = TextEdit.new()
	text_edit.text = str(p.get("value", p.get("default", "")))
	text_edit.size_flags_vertical = Control.SIZE_EXPAND_FILL
	text_edit.add_theme_font_size_override("font_size", 14)
	# 开启代码语法高亮的常用设置
	text_edit.syntax_highlighter = CodeHighlighter.new()
	text_edit.draw_tabs = true
	vbox.add_child(text_edit)
	
	var save_btn = Button.new()
	save_btn.text = "💾 保存并关闭"
	save_btn.custom_minimum_size.y = 40
	save_btn.pressed.connect(func():
		p["value"] = text_edit.text
		win.queue_free()
		parameter_changed.emit()
	)
	vbox.add_child(save_btn)
	
	win.close_requested.connect(func(): win.queue_free())
	add_child(win)
	win.popup_centered()
