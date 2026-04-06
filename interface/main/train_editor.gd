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
	
	graph_edit.popup_request.connect(_on_graph_edit_popup_request)
	graph_edit.delete_nodes_request.connect(func(n): 
		for nn in n: 
			var nd = graph_edit.get_node(NodePath(nn))
			if nd: _safe_delete_node(nd)
	)
	graph_edit.connection_request.connect(_on_connection_request)
	graph_edit.disconnection_request.connect(_on_disconnection_request)
	GlobalData.library_updated.connect(func(): load_graph_state())
	if not GlobalData.node_library.is_empty(): 
		load_graph_state()

func _process(_delta):
	for conn in graph_edit.get_connection_list():
		var ck = str(conn["from_node"]) + "->" + str(conn["to_node"])
		if shape_labels.has(ck):
			var na = graph_edit.get_node(NodePath(str(conn["from_node"]))) as GraphNode
			var nb = graph_edit.get_node(NodePath(str(conn["to_node"]))) as GraphNode
			if na and nb:
				var ml = ((na.position_offset + na.get_output_port_position(conn["from_port"]) + nb.position_offset + nb.get_input_port_position(conn["to_port"])) / 2.0 * graph_edit.zoom) - graph_edit.scroll_offset
				var lbl = shape_labels[ck] as Label
				lbl.reset_size()
				lbl.position = ml - (lbl.size / 2.0)

func import_model_as_group(model_data: Dictionary, model_name: String):
	pass # 扁平架构不再导入单独图表

func apply_shape_results(p: Dictionary):
	for ck in p.get("shapes", {}).keys():
		if shape_labels.has(ck):
			var lbl = shape_labels[ck] as Label
			var res = str(p["shapes"][ck])
			if res.length() > 60: res = res.substr(0, 57) + "..."
			if "Error" in res or "冲突" in res: 
				lbl.text = "✖ " + res
				lbl.label_settings.font_color = Color(1, 0.3, 0.3)
			else: 
				lbl.text = " " + res + " "
				lbl.label_settings.font_color = Color(0.4, 1.0, 0.6)

func save_current_graph_state():
	var data = {"nodes": {}, "connections": []}
	for c in graph_edit.get_children():
		if c is GraphNode and c.has_method("get_current_params"): 
			data["nodes"][c.name] = {
				"type": c.config.name, 
				"params": c.get_current_params(), 
				"inputs": c.config.get("inputs", []), 
				"pos_x": c.position_offset.x, 
				"pos_y": c.position_offset.y
			}
	for conn in graph_edit.get_connection_list(): 
		data["connections"].append({
			"from": conn["from_node"], 
			"from_port": conn["from_port"], 
			"to": conn["to_node"], 
			"to_port": conn["to_port"]
		})
	GlobalData.project_graphs["train"] = data

func load_graph_state():
	graph_edit.clear_connections()
	for c in graph_edit.get_children(): 
		if c is GraphNode or c is Label: c.queue_free()
	shape_labels.clear()
	await get_tree().process_frame
	
	var data = GlobalData.project_graphs.get("train", {})
	if data.get("nodes", {}).is_empty():
		if GlobalData.node_library.has("Training (训练配置)"):
			var d = create_node("Training (训练配置)", "Dataset Loader", Vector2(50, 100))
			d.name = "Dataset"
			var c = create_node("Training (训练配置)", "Training Config", Vector2(350, 100))
			c.name = "Config"
			var t = create_node("Training (训练配置)", "Target Loader", Vector2(50, 350))
			t.name = "Target"
			var l = create_node("Losses (损失函数)", "CrossEntropyLoss", Vector2(1000, 250))
			l.name = "Loss"
			var o = create_node("Optimizers (优化器)", "Adam", Vector2(1350, 250))
			o.name = "Optimizer"
			await get_tree().process_frame
			_on_connection_request(d.name, 0, c.name, 0)
			_on_connection_request(t.name, 0, l.name, 1)
			_on_connection_request(l.name, 0, o.name, 0)
	else:
		for nn in data["nodes"].keys():
			var nd = data["nodes"][nn]
			for cat in GlobalData.node_library.keys():
				if GlobalData.node_library[cat].has(nd["type"]):
					var conf = GlobalData.node_library[cat][nd["type"]].duplicate(true)
					for p in conf.params: 
						if nd.get("params", {}).has(p.name): p["value"] = nd["params"][p.name]["value"]
					var n = _create_node(conf, Vector2(nd.get("pos_x", 0), nd.get("pos_y", 0)))
					n.name = nn
					break
		await get_tree().process_frame
		for c in data["connections"]: 
			_on_connection_request(c["from"], c["from_port"], c["to"], c["to_port"])
	_request_shape_inference()

func _request_shape_inference(): 
	save_current_graph_state()

func _on_connection_request(fn: StringName, fp: int, tn: StringName, tp: int):
	for conn in graph_edit.get_connection_list(): 
		if conn["to_node"] == tn and conn["to_port"] == tp: 
			graph_edit.disconnect_node(conn["from_node"], conn["from_port"], conn["to_node"], conn["to_port"])
	graph_edit.connect_node(fn, fp, tn, tp)

func _on_disconnection_request(fn: StringName, fp: int, tn: StringName, tp: int): 
	graph_edit.disconnect_node(fn, fp, tn, tp)
	
func _create_node(c: Dictionary, p: Vector2) -> GraphNode:
	var nn = DynamicNodeScene.instantiate()
	graph_edit.add_child(nn)
	nn.setup_node(c)
	nn.position_offset = p
	nn.gui_input.connect(_on_node_gui_input.bind(nn))
	nn.parameter_changed.connect(_request_shape_inference)
	return nn
	
func create_node(c, t, p): 
	return _create_node(GlobalData.node_library[c][t].duplicate(true), p)

func _build_action_menu(): 
	action_menu.clear()
	action_menu.add_item("删除", 2)
	action_menu.id_pressed.connect(func(id): 
		for n in graph_edit.get_children(): 
			if n is GraphNode and n.selected: _safe_delete_node(n)
	)
	
func _on_node_gui_input(e, n): 
	if e is InputEventMouseButton and e.button_index == MOUSE_BUTTON_RIGHT and e.pressed: 
		action_menu.position = get_viewport().get_mouse_position()
		action_menu.popup()
		n.accept_event()
		
func _safe_delete_node(node: GraphNode): 
	var nn = node.name
	for c in graph_edit.get_connection_list(): 
		if c["from_node"] == nn or c["to_node"] == nn: 
			_on_disconnection_request(c["from_node"], c["from_port"], c["to_node"], c["to_port"])
	node.queue_free()

func _setup_search_menu(): 
	search_popup = PopupPanel.new()
	var v = VBoxContainer.new()
	search_box = LineEdit.new()
	search_box.text_changed.connect(_populate_tree)
	v.add_child(search_box)
	node_tree = Tree.new()
	node_tree.hide_root = true
	node_tree.item_activated.connect(_on_tree_item_chosen)
	v.add_child(node_tree)
	search_popup.add_child(v)
	add_child(search_popup) 
	
func _on_graph_edit_popup_request(pos): 
	last_mouse_position = (pos + graph_edit.scroll_offset) / graph_edit.zoom
	_populate_tree("")
	search_popup.position = get_viewport().get_mouse_position()
	search_popup.popup()
	search_box.clear()
	search_box.grab_focus() 
	
func _populate_tree(ft: String): 
	node_tree.clear()
	var root = node_tree.create_item()
	ft = ft.to_lower()
	for cat in GlobalData.node_library.keys(): 
		var ci = node_tree.create_item(root)
		ci.set_text(0, cat)
		var vis = false
		for nn in GlobalData.node_library[cat].keys(): 
			if ft == "" or ft in nn.to_lower() or ft in cat.to_lower(): 
				var i = node_tree.create_item(ci)
				i.set_text(0, nn)
				i.set_metadata(0, {"c": cat, "n": nn})
				vis = true
		if not vis: 
			ci.free()
		else: 
			if ft == "":
				ci.collapsed = true
			else:
				ci.collapsed = false
				
func _on_tree_item_chosen(): 
	var s = node_tree.get_selected()
	if s and s.get_metadata(0): 
		_create_node(GlobalData.node_library[s.get_metadata(0)["c"]][s.get_metadata(0)["n"]], last_mouse_position)
		search_popup.hide()
