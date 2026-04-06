extends Control

@onready var graph_edit = $GraphEdit
@onready var action_menu = $ActionMenu 

const DynamicNodeScene = preload("res://torchnode/dynamic_torch_node.tscn") 
var last_mouse_position: Vector2 = Vector2.ZERO
var clipboard_nodes: Array = [] 
var search_popup: PopupPanel
var search_box: LineEdit
var node_tree: Tree
var shape_labels: Dictionary = {}

func _ready():
	set_anchors_and_offsets_preset(PRESET_FULL_RECT)
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
	
	GlobalData.library_updated.connect(func(): load_graph_state())
	if not GlobalData.node_library.is_empty():
		load_graph_state()

func _process(_delta):
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

func _unhandled_input(event: InputEvent):
	if event is InputEventKey and event.pressed and not event.is_echo():
		var focus_owner = get_viewport().gui_get_focus_owner()
		if focus_owner is LineEdit or focus_owner is TextEdit:
			return 
		var is_cmd = event.ctrl_pressed or event.meta_pressed 
		var sel = _get_selected_nodes()
		match event.keycode:
			KEY_C: 
				if is_cmd and not sel.is_empty(): 
					_copy_selected_nodes(sel)
					get_viewport().set_input_as_handled()
			KEY_X: 
				if is_cmd and not sel.is_empty(): 
					_copy_selected_nodes(sel)
					for n in sel: _safe_delete_node(n)
					get_viewport().set_input_as_handled()
			KEY_V: 
				if is_cmd and not clipboard_nodes.is_empty(): 
					_paste_nodes_from_clipboard()
					get_viewport().set_input_as_handled()
			KEY_A: 
				if is_cmd: 
					for n in graph_edit.get_children(): 
						if n is GraphNode: n.selected = true
					get_viewport().set_input_as_handled()
			KEY_D: 
				if is_cmd and not sel.is_empty(): 
					_copy_selected_nodes(sel)
					_paste_nodes_from_clipboard(Vector2(20, 20))
					get_viewport().set_input_as_handled()
			KEY_DELETE, KEY_BACKSPACE: 
				if not sel.is_empty(): 
					for n in sel: _safe_delete_node(n)
					get_viewport().set_input_as_handled()

func _get_selected_nodes() -> Array:
	var sel = []
	for n in graph_edit.get_children():
		if n is GraphNode and n.selected:
			sel.append(n)
	return sel

func _copy_selected_nodes(nodes: Array):
	clipboard_nodes.clear()
	if nodes.is_empty(): return
	var base_pos = nodes[0].position_offset
	for n in nodes: 
		clipboard_nodes.append({
			"config": n.config.duplicate(true), 
			"offset": n.position_offset - base_pos
		})

func _paste_nodes_from_clipboard(extra_offset: Vector2 = Vector2.ZERO):
	if clipboard_nodes.is_empty(): return
	for n in graph_edit.get_children(): 
		if n is GraphNode: n.selected = false 
	var pb = (graph_edit.get_local_mouse_position() + graph_edit.scroll_offset) / graph_edit.zoom + extra_offset
	for item in clipboard_nodes:
		var nn = _create_node(item["config"].duplicate(true), pb + item["offset"])
		nn.selected = true
	_request_shape_inference()

func _safe_delete_node(node: GraphNode):
	var nn = node.name
	for conn in graph_edit.get_connection_list():
		if conn["from_node"] == nn or conn["to_node"] == nn:
			_on_disconnection_request(conn["from_node"], conn["from_port"], conn["to_node"], conn["to_port"])
	node.queue_free()

func apply_shape_results(payload: Dictionary):
	var shape_data = payload.get("shapes", {})
	for conn_key in shape_data.keys():
		if shape_labels.has(conn_key):
			var lbl = shape_labels[conn_key] as Label
			var result = str(shape_data[conn_key])
			if result.length() > 60: result = result.substr(0, 57) + "..."
			if "Error" in result or "冲突" in result or "错误" in result: 
				lbl.text = "✖ " + result
				lbl.label_settings.font_color = Color(1, 0.3, 0.3)
			else: 
				lbl.text = " " + result + " "
				lbl.label_settings.font_color = Color(0.4, 1.0, 0.6)
	var updated_params = payload.get("updated_params", {})
	for child in graph_edit.get_children(): 
		if child is GraphNode and child.has_method("reset_auto_params"): child.reset_auto_params()
	for nn in updated_params.keys():
		var node = graph_edit.get_node_or_null(NodePath(nn))
		if node and node is GraphNode and node.has_method("apply_auto_params"): 
			node.apply_auto_params(updated_params[nn])

func save_current_graph_state():
	var data = {"nodes": {}, "connections": []}
	for child in graph_edit.get_children():
		if child is GraphNode and child.has_method("get_current_params"):
			data["nodes"][child.name] = {
				"type": child.config.name, 
				"params": child.get_current_params(), 
				"inputs": child.config.get("inputs", []), 
				"pos_x": child.position_offset.x, 
				"pos_y": child.position_offset.y
			}
	for conn in graph_edit.get_connection_list():
		data["connections"].append({
			"from": conn["from_node"], 
			"from_port": conn["from_port"], 
			"to": conn["to_node"], 
			"to_port": conn["to_port"]
		})
	GlobalData.project_graphs["main"] = data

func load_graph_state():
	graph_edit.clear_connections()
	for child in graph_edit.get_children(): 
		if child is GraphNode or child is Label: child.queue_free()
	shape_labels.clear()
	await get_tree().process_frame
	var data = GlobalData.project_graphs.get("main", {})
	if data.get("nodes", {}).is_empty():
		pass
	else:
		for nn in data["nodes"].keys():
			var n_data = data["nodes"][nn]
			var cat = _find_category_by_type(n_data["type"])
			if cat != "":
				var conf = GlobalData.node_library[cat][n_data["type"]].duplicate(true)
				if n_data.has("params") and conf.has("params"):
					for p in conf.params: 
						if n_data["params"].has(p.name): p["value"] = n_data["params"][p.name]["value"]
				var n = create_node_from_config(conf, Vector2(n_data.get("pos_x", 0), n_data.get("pos_y", 0)))
				n.name = nn
		await get_tree().process_frame
		for c in data["connections"]: 
			_on_connection_request(c["from"], c["from_port"], c["to"], c["to_port"])
	_request_shape_inference()

func _request_shape_inference():
	save_current_graph_state()
	if GlobalData.socket.get_ready_state() == WebSocketPeer.STATE_OPEN: 
		GlobalData.socket.send_text("INFER_SHAPES:" + JSON.stringify(GlobalData.project_graphs))

func _on_connection_request(fn: StringName, fp: int, tn: StringName, tp: int):
	for conn in graph_edit.get_connection_list(): 
		if conn["from_node"] == fn and conn["from_port"] == fp and conn["to_node"] == tn and conn["to_port"] == tp: 
			return 
	for conn in graph_edit.get_connection_list():
		if conn["to_node"] == tn and conn["to_port"] == tp:
			graph_edit.disconnect_node(conn["from_node"], conn["from_port"], conn["to_node"], conn["to_port"])
			var ok = str(conn["from_node"]) + "->" + str(conn["to_node"])
			if shape_labels.has(ok): 
				shape_labels[ok].queue_free()
				shape_labels.erase(ok)
			break 
			
	graph_edit.connect_node(fn, fp, tn, tp)
	var ck = str(fn) + "->" + str(tn)
	if not shape_labels.has(ck):
		var lbl = Label.new()
		lbl.text = " 计算中... "
		lbl.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
		var s = LabelSettings.new()
		s.font_size = 13
		s.font_color = Color(0.8, 0.8, 0.8)
		lbl.label_settings = s
		var sty = StyleBoxFlat.new()
		sty.bg_color = Color(0.1, 0.1, 0.15, 0.9)
		sty.set_corner_radius_all(6)
		sty.content_margin_left = 8
		sty.content_margin_right = 8
		sty.border_width_left = 1
		sty.border_color = Color(0.3, 0.3, 0.4)
		lbl.add_theme_stylebox_override("normal", sty) 
		graph_edit.add_child(lbl)
		shape_labels[ck] = lbl
	_request_shape_inference()

func _on_disconnection_request(fn: StringName, fp: int, tn: StringName, tp: int):
	graph_edit.disconnect_node(fn, fp, tn, tp)
	var ck = str(fn) + "->" + str(tn)
	if shape_labels.has(ck): 
		shape_labels[ck].queue_free()
		shape_labels.erase(ck)
	_request_shape_inference()

func _find_category_by_type(nt: String) -> String:
	for cat in GlobalData.node_library.keys(): 
		if GlobalData.node_library[cat].has(nt): return cat
	return ""

func _create_node(c: Dictionary, p: Vector2) -> GraphNode: 
	return create_node_from_config(c.duplicate(true), p)

func create_node(c, t, p) -> GraphNode: 
	return create_node_from_config(GlobalData.node_library[c][t].duplicate(true), p)

func create_node_from_config(c: Dictionary, p: Vector2) -> GraphNode:
	var nn = DynamicNodeScene.instantiate()
	graph_edit.add_child(nn)
	nn.setup_node(c)
	nn.position_offset = p
	nn.gui_input.connect(_on_node_gui_input.bind(nn))
	nn.parameter_changed.connect(_request_shape_inference)
	return nn

func _build_action_menu():
	action_menu.clear()
	action_menu.add_item("复制 (Copy)", 0)
	action_menu.add_item("剪切 (Cut)", 1)
	action_menu.add_separator()
	action_menu.add_item("删除 (Delete)", 2)
	action_menu.id_pressed.connect(func(id: int): 
		var sel = _get_selected_nodes()
		if sel.is_empty(): return
		match id:
			0: _copy_selected_nodes(sel)
			1: 
				_copy_selected_nodes(sel)
				for n in sel: _safe_delete_node(n)
			2: 
				for n in sel: _safe_delete_node(n)
	)

func _on_node_gui_input(event: InputEvent, node: GraphNode):
	if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_RIGHT and event.pressed:
		if not node.selected: 
			for n in graph_edit.get_children(): 
				if n is GraphNode: n.selected = (n == node)
		action_menu.position = get_viewport().get_mouse_position()
		action_menu.popup()
		node.accept_event()

func _on_delete_nodes_request(nodes: Array[StringName]): 
	for nn in nodes: 
		var n = graph_edit.get_node(NodePath(nn))
		if n: _safe_delete_node(n)

func _setup_search_menu():
	search_popup = PopupPanel.new()
	var vbox = VBoxContainer.new()
	search_box = LineEdit.new()
	search_box.placeholder_text = "搜索 PyTorch 节点..."
	search_box.text_changed.connect(_populate_tree)
	vbox.add_child(search_box)
	
	node_tree = Tree.new()
	node_tree.custom_minimum_size = Vector2(320, 450)
	node_tree.hide_root = true 
	node_tree.item_selected.connect(_on_tree_item_chosen)
	node_tree.item_activated.connect(_on_tree_item_chosen) 
	vbox.add_child(node_tree)
	search_popup.add_child(vbox)
	add_child(search_popup) 

func _on_graph_edit_popup_request(pos: Vector2):
	last_mouse_position = (pos + graph_edit.scroll_offset) / graph_edit.zoom
	_populate_tree("") 
	search_popup.position = get_viewport().get_mouse_position()
	search_popup.popup()
	search_box.clear()
	search_box.grab_focus() 

func _populate_tree(filter_text: String):
	node_tree.clear()
	var root = node_tree.create_item()
	if filter_text == "" and not clipboard_nodes.is_empty():
		var paste_item = node_tree.create_item(root)
		paste_item.set_text(0, "📋 粘贴 " + str(clipboard_nodes.size()) + " 个节点")
		paste_item.set_metadata(0, {"is_paste": true})
		paste_item.set_custom_color(0, Color(1, 0.8, 0.2))
		
	filter_text = filter_text.to_lower()
	for cat in GlobalData.node_library.keys():
		var cat_item = node_tree.create_item(root)
		cat_item.set_text(0, cat)
		cat_item.set_selectable(0, false)
		cat_item.set_custom_bg_color(0, Color(0.15, 0.15, 0.15))
		var has_visible_children = false
		for node_name in GlobalData.node_library[cat].keys():
			if filter_text == "" or filter_text in node_name.to_lower() or filter_text in cat.to_lower():
				var item = node_tree.create_item(cat_item)
				item.set_text(0, node_name)
				item.set_metadata(0, {"category": cat, "name": node_name})
				has_visible_children = true
				
		if not has_visible_children: 
			cat_item.free()
		else: 
			if filter_text == "":
				cat_item.collapsed = true
			else:
				cat_item.collapsed = false

func _on_tree_item_chosen():
	var selected_item = node_tree.get_selected()
	if not selected_item: return
	var meta = selected_item.get_metadata(0)
	if meta == null: return
	
	if meta.has("is_paste"): 
		_paste_nodes_from_clipboard(last_mouse_position - (graph_edit.scroll_offset / graph_edit.zoom))
	else: 
		_create_node(GlobalData.node_library[meta["category"]][meta["name"]], last_mouse_position)
		
	selected_item.deselect(0)
	search_popup.hide()
