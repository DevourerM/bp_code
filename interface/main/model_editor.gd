extends Control

@onready var graph_edit = $GraphEdit
@onready var action_menu = $ActionMenu 

# 面包屑依然属于特定编辑器内部的导航
var breadcrumb_bar: PanelContainer# 如果你在编辑器做好了就直接绑，否则保留代码创建
var breadcrumb_container: HBoxContainer

const DynamicNodeScene = preload("res://torchnode/dynamic_torch_node.tscn") 

var last_mouse_position: Vector2 = Vector2.ZERO
var clipboard_config: Dictionary = {}
var search_popup: PopupPanel; var search_box: LineEdit; var node_tree: Tree
var shape_labels: Dictionary = {}

func _ready():
	# UI 剥离后，我们必须让这个控件充满父容器(App的EditorContainer)
	set_anchors_and_offsets_preset(PRESET_FULL_RECT)
	
	_setup_breadcrumbs_ui()
	_setup_search_menu()
	_build_action_menu()
	
	graph_edit.panning_scheme = GraphEdit.SCROLL_ZOOMS
	graph_edit.right_disconnects = true 
	graph_edit.connection_lines_thickness = 4 
	graph_edit.connection_lines_antialiased = true 
	
	graph_edit.popup_request.connect(_on_graph_edit_popup_request)
	graph_edit.delete_nodes_request.connect(_on_delete_nodes_request)
	graph_edit.connection_request.connect(_on_connection_request)
	graph_edit.disconnection_request.connect(_on_disconnection_request)
	
	GlobalData.library_updated.connect(func(): load_graph_state("main"))
	if not GlobalData.node_library.is_empty():
		load_graph_state(GlobalData.get_current_graph_id())

func _process(_delta):
	# 注意：WebSocket 轮询已经搬到 app.gd 了，这里只负责更新标签的 UI 坐标
	for conn in graph_edit.get_connection_list():
		var conn_key = str(conn["from_node"]) + "->" + str(conn["to_node"])
		if shape_labels.has(conn_key):
			var node_a = graph_edit.get_node(NodePath(str(conn["from_node"]))) as GraphNode
			var node_b = graph_edit.get_node(NodePath(str(conn["to_node"]))) as GraphNode
			if node_a and node_b:
				var pos_a = node_a.position_offset + node_a.get_output_port_position(conn["from_port"])
				var pos_b = node_b.position_offset + node_b.get_input_port_position(conn["to_port"])
				var mid_local = ((pos_a + pos_b) / 2.0 * graph_edit.zoom) - graph_edit.scroll_offset
				var lbl = shape_labels[conn_key] as Label
				lbl.reset_size() 
				lbl.position = mid_local - (lbl.size / 2.0)

# ==========================================
# 开放给 App 调用的公开接口 (API)
# ==========================================
func apply_shape_results(payload: Dictionary):
	var shape_data = payload.get("shapes", {})
	for conn_key in shape_data.keys():
		if shape_labels.has(conn_key):
			var lbl = shape_labels[conn_key] as Label
			var result = str(shape_data[conn_key])
			if result.length() > 60: result = result.substr(0, 57) + "..."
			if "Error" in result or "冲突" in result or "错误" in result:
				lbl.text = "✖ " + result; lbl.label_settings.font_color = Color(1, 0.3, 0.3)
			else:
				lbl.text = " " + result + " "; lbl.label_settings.font_color = Color(0.4, 1.0, 0.6)
				
	var updated_params = payload.get("updated_params", {})
	for child in graph_edit.get_children():
		if child is GraphNode and child.has_method("reset_auto_params"): child.reset_auto_params()
	for node_name in updated_params.keys():
		var node = graph_edit.get_node_or_null(NodePath(node_name))
		if node and node is GraphNode and node.has_method("apply_auto_params"):
			node.apply_auto_params(updated_params[node_name])

func save_current_graph_state():
	var current_id = GlobalData.get_current_graph_id()
	var data = {"nodes": {}, "connections": []}
	for child in graph_edit.get_children():
		if child is GraphNode and child.has_method("get_current_params"):
			data["nodes"][child.name] = {
				"type": child.config.name, "params": child.get_current_params(),
				"inputs": child.config.get("inputs", []), "pos_x": child.position_offset.x, "pos_y": child.position_offset.y
			}
	for conn in graph_edit.get_connection_list():
		data["connections"].append({"from": conn["from_node"], "from_port": conn["from_port"], "to": conn["to_node"], "to_port": conn["to_port"]})
	GlobalData.project_graphs[current_id] = data

func load_graph_state(graph_id: String, expected_input_count: int = 1):
	graph_edit.clear_connections()
	for child in graph_edit.get_children():
		if child is GraphNode or child is Label: child.queue_free()
	shape_labels.clear()
	await get_tree().process_frame
	
	var data = GlobalData.project_graphs.get(graph_id, {})
	var sys_cat = "System (系统节点)"
	
	if data.get("nodes", {}).is_empty():
		if graph_id == "main":
			var in_n = create_node(sys_cat, "Data Input", Vector2(100, 200))
			in_n.name = "Data Input"; in_n.title = "Data Input"
			var out_n = create_node(sys_cat, "Data Output", Vector2(800, 200))
			out_n.name = "Data Output"; out_n.title = "Data Output"
			_on_connection_request(in_n.name, 0, out_n.name, 0)
		else:
			for i in range(expected_input_count):
				var in_name = "input" + str(i)
				var in_n = create_node(sys_cat, "Data Input", Vector2(100, 200 + i * 150))
				in_n.name = in_name; in_n.title = in_name
			var out_n = create_node(sys_cat, "Data Output", Vector2(800, 200))
			out_n.name = "output0"; out_n.title = "output0"
	else:
		for node_name in data["nodes"].keys():
			var n_data = data["nodes"][node_name]
			var cat = _find_category_by_type(n_data["type"])
			if cat != "":
				var conf = GlobalData.node_library[cat][n_data["type"]].duplicate(true)
				if n_data.has("params") and conf.has("params"):
					for p in conf.params:
						if n_data["params"].has(p.name): p["value"] = n_data["params"][p.name]["value"]
				var n = create_node_from_config(conf, Vector2(n_data.get("pos_x", 0), n_data.get("pos_y", 0)))
				n.name = node_name
				if node_name.begins_with("input") or node_name.begins_with("output"): n.title = node_name
				
		await get_tree().process_frame
		for c in data["connections"]:
			_on_connection_request(c["from"], c["from_port"], c["to"], c["to_port"])
			
		if graph_id != "main":
			for i in range(expected_input_count):
				var in_name = "input" + str(i)
				if not graph_edit.has_node(NodePath(in_name)):
					var n = create_node(sys_cat, "Data Input", Vector2(100, 200 + i * 150))
					n.name = in_name; n.title = in_name
			for child in graph_edit.get_children():
				if child is GraphNode and child.config.name == "Data Input" and child.name.begins_with("input"):
					var idx_str = child.name.replace("input", "")
					if idx_str.is_valid_int() and int(idx_str) >= expected_input_count:
						_safe_delete_node(child)
						
	_refresh_breadcrumbs()
	_request_shape_inference()

# ==========================================
# 以下均为内部逻辑 (连线推导/面包屑/搜索树)
# ==========================================
func _request_shape_inference():
	save_current_graph_state()
	if GlobalData.socket.get_ready_state() == WebSocketPeer.STATE_OPEN:
		GlobalData.socket.send_text("INFER_SHAPES:" + JSON.stringify(GlobalData.project_graphs))

func _on_enter_subgraph_requested(node_name: String):
	var node = graph_edit.get_node_or_null(NodePath(node_name))
	var expected_count = 1
	if node:
		var params = node.get_current_params()
		if params.has("input_count"): expected_count = int(float(params["input_count"]["value"]))
		
	save_current_graph_state()
	GlobalData.current_path.append(node_name)
	load_graph_state(node_name, expected_count)

func _go_to_breadcrumb_level(level_index: int):
	if level_index == GlobalData.current_path.size() - 1: return
	save_current_graph_state()
	GlobalData.current_path = GlobalData.current_path.slice(0, level_index + 1)
	load_graph_state(GlobalData.get_current_graph_id())

func _on_connection_request(from_node: StringName, from_port: int, to_node: StringName, to_port: int):
	for conn in graph_edit.get_connection_list():
		if conn["from_node"] == from_node and conn["from_port"] == from_port and conn["to_node"] == to_node and conn["to_port"] == to_port: return 
	for conn in graph_edit.get_connection_list():
		if conn["to_node"] == to_node and conn["to_port"] == to_port:
			graph_edit.disconnect_node(conn["from_node"], conn["from_port"], conn["to_node"], conn["to_port"])
			var old_key = str(conn["from_node"]) + "->" + str(conn["to_node"])
			if shape_labels.has(old_key): shape_labels[old_key].queue_free(); shape_labels.erase(old_key)
			break 
			
	graph_edit.connect_node(from_node, from_port, to_node, to_port)
	var conn_key = str(from_node) + "->" + str(to_node)
	if not shape_labels.has(conn_key):
		var lbl = Label.new(); lbl.text = " 计算中... "; lbl.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
		var settings = LabelSettings.new(); settings.font_size = 13; settings.font_color = Color(0.8, 0.8, 0.8)
		lbl.label_settings = settings
		var style = StyleBoxFlat.new(); style.bg_color = Color(0.1, 0.1, 0.15, 0.9); style.set_corner_radius_all(6)
		style.content_margin_left = 8; style.content_margin_right = 8; style.content_margin_top = 4; style.content_margin_bottom = 4
		style.border_width_left = 1; style.border_width_right = 1; style.border_width_top = 1; style.border_width_bottom = 1
		style.border_color = Color(0.3, 0.3, 0.4)
		lbl.add_theme_stylebox_override("normal", style) 
		graph_edit.add_child(lbl)
		shape_labels[conn_key] = lbl
	_request_shape_inference()

func _on_disconnection_request(from_node: StringName, from_port: int, to_node: StringName, to_port: int):
	graph_edit.disconnect_node(from_node, from_port, to_node, to_port)
	var conn_key = str(from_node) + "->" + str(to_node)
	if shape_labels.has(conn_key):
		shape_labels[conn_key].queue_free()
		shape_labels.erase(conn_key)
	_request_shape_inference()

func _find_category_by_type(node_type: String) -> String:
	for cat in GlobalData.node_library.keys():
		if GlobalData.node_library[cat].has(node_type): return cat
	return ""

func _create_node(config: Dictionary, pos: Vector2) -> GraphNode:
	return create_node_from_config(config.duplicate(true), pos)

func create_node(category, type, pos) -> GraphNode:
	return create_node_from_config(GlobalData.node_library[category][type].duplicate(true), pos)

func create_node_from_config(config: Dictionary, pos: Vector2) -> GraphNode:
	var new_node = DynamicNodeScene.instantiate()
	graph_edit.add_child(new_node)
	new_node.setup_node(config)
	new_node.position_offset = pos
	new_node.gui_input.connect(_on_node_gui_input.bind(new_node))
	new_node.parameter_changed.connect(_request_shape_inference)
	if new_node.has_signal("enter_subgraph_requested"):
		new_node.enter_subgraph_requested.connect(_on_enter_subgraph_requested)
	return new_node

func _setup_breadcrumbs_ui():
	if not breadcrumb_bar:
		breadcrumb_bar = PanelContainer.new()
		var style = StyleBoxFlat.new(); style.bg_color = Color(0.1, 0.1, 0.1, 0.8)
		breadcrumb_bar.add_theme_stylebox_override("panel", style)
		breadcrumb_container = HBoxContainer.new()
		breadcrumb_bar.add_child(breadcrumb_container)
		add_child(breadcrumb_bar)
		breadcrumb_bar.set_anchors_and_offsets_preset(PRESET_TOP_WIDE)
		breadcrumb_bar.custom_minimum_size.y = 40
	elif breadcrumb_bar.get_child_count() > 0:
		breadcrumb_container = breadcrumb_bar.get_child(0)

func _refresh_breadcrumbs():
	for c in breadcrumb_container.get_children(): c.queue_free()
	var path = GlobalData.current_path
	for i in range(path.size()):
		var btn = Button.new()
		# 【修改】：如果是根目录，显示场景名而不是 main
		btn.text = "Model" if str(path[i]) == "main" else str(path[i])
		btn.add_theme_font_size_override("font_size", 16)
		btn.pressed.connect(func(): _go_to_breadcrumb_level(i))
		breadcrumb_container.add_child(btn)
		if i < path.size() - 1:
			var sep = Label.new(); sep.text = "  >>  "
			breadcrumb_container.add_child(sep)

func _setup_search_menu():
	search_popup = PopupPanel.new()
	var vbox = VBoxContainer.new()
	search_box = LineEdit.new(); search_box.placeholder_text = "搜索 PyTorch 节点..."
	search_box.text_changed.connect(_on_search_text_changed)
	vbox.add_child(search_box)
	node_tree = Tree.new(); node_tree.custom_minimum_size = Vector2(320, 450); node_tree.hide_root = true 
	node_tree.item_selected.connect(_on_tree_item_chosen)
	node_tree.item_activated.connect(_on_tree_item_chosen) 
	vbox.add_child(node_tree); search_popup.add_child(vbox); add_child(search_popup) 

func _populate_tree(filter_text: String):
	node_tree.clear()
	var root = node_tree.create_item()
	if filter_text == "" and not clipboard_config.is_empty():
		var paste_item = node_tree.create_item(root)
		paste_item.set_text(0, "📋 粘贴 (Paste)"); paste_item.set_metadata(0, {"is_paste": true}); paste_item.set_custom_color(0, Color(1, 0.8, 0.2)) 
	filter_text = filter_text.to_lower()
	for category in GlobalData.node_library.keys():
		var cat_item = node_tree.create_item(root)
		cat_item.set_text(0, category); cat_item.set_selectable(0, false); cat_item.set_custom_bg_color(0, Color(0.15, 0.15, 0.15))
		var has_visible_children = false
		for node_name in GlobalData.node_library[category].keys():
			if filter_text == "" or filter_text in node_name.to_lower() or filter_text in category.to_lower():
				var item = node_tree.create_item(cat_item)
				item.set_text(0, node_name); item.set_metadata(0, {"category": category, "name": node_name})
				item.set_tooltip_text(0, "[ " + node_name + " ]\n" + GlobalData.node_library[category][node_name].get("description", "暂无说明文档"))
				has_visible_children = true
		if not has_visible_children: cat_item.free()
		elif filter_text != "": cat_item.collapsed = false 

func _on_search_text_changed(new_text: String): _populate_tree(new_text)

func _on_tree_item_chosen():
	var selected_item = node_tree.get_selected()
	if not selected_item: return
	var meta = selected_item.get_metadata(0)
	if meta == null: return
	if meta.has("is_paste"): _create_node(clipboard_config.duplicate(true), last_mouse_position)
	else: _create_node(GlobalData.node_library[meta["category"]][meta["name"]], last_mouse_position)
	selected_item.deselect(0); search_popup.hide() 

func _on_graph_edit_popup_request(pos: Vector2):
	last_mouse_position = (pos + graph_edit.scroll_offset) / graph_edit.zoom
	_populate_tree("") 
	search_popup.position = get_viewport().get_mouse_position(); search_popup.popup()
	search_box.clear(); search_box.grab_focus() 

func _build_action_menu():
	action_menu.clear()
	action_menu.add_item("复制 (Copy)", 0); action_menu.add_item("剪切 (Cut)", 1); action_menu.add_separator(); action_menu.add_item("删除 (Delete)", 2)
	action_menu.id_pressed.connect(_on_action_menu_id_pressed)

func _on_node_gui_input(event: InputEvent, node: GraphNode):
	if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_RIGHT and event.pressed:
		for n in graph_edit.get_children():
			if n is GraphNode: n.selected = (n == node)
		action_menu.position = get_viewport().get_mouse_position(); action_menu.popup()
		node.accept_event()

func _on_action_menu_id_pressed(id: int):
	var selected_node = null
	for n in graph_edit.get_children():
		if n is GraphNode and n.selected: selected_node = n; break
	if not selected_node: return
	match id:
		0: clipboard_config = selected_node.config.duplicate(true)
		1: clipboard_config = selected_node.config.duplicate(true); _safe_delete_node(selected_node)
		2: _safe_delete_node(selected_node)

func _safe_delete_node(node: GraphNode):
	var node_name = node.name
	for conn in graph_edit.get_connection_list():
		if conn["from_node"] == node_name or conn["to_node"] == node_name:
			_on_disconnection_request(conn["from_node"], conn["from_port"], conn["to_node"], conn["to_port"])
	node.queue_free()

func _on_delete_nodes_request(nodes: Array[StringName]):
	for node_name in nodes:
		var node = graph_edit.get_node(NodePath(node_name))
		if node: _safe_delete_node(node)
