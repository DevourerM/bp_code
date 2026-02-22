extends Control

@onready var editor_container = $EditorContainer
@onready var save_dialog: FileDialog = $Dialogs/SaveDialog
@onready var load_dialog: FileDialog = $Dialogs/LoadDialog
@onready var import_dialog: FileDialog = $Dialogs/ImportDialog
@onready var export_dialog: FileDialog = $Dialogs/ExportDialog
@onready var export_window: Window = $Dialogs/ExportWindow
@onready var export_class_input: LineEdit = $Dialogs/ExportWindow/VBoxContainer/ClassInput
@onready var message_dialog: AcceptDialog = $Dialogs/MessageDialog

var current_editor: Node = null
var current_tab: String = "model" 

# 【核心修复】：为三个模块准备独立的内存工作区，防止节点互相串台！
var workspaces = {
	"model": {}, "train": {}, "test": {}
}

var tab_extensions = {
	"model": { "filter": "*.bpnn", "ext": ".bpnn", "desc": "Model Blueprint" },
	"train": { "filter": "*.bptr", "ext": ".bptr", "desc": "Train Blueprint" },
	"test":  { "filter": "*.bpte", "ext": ".bpte", "desc": "Test Blueprint" }
}

func _ready():
	_setup_dialogs()
	_connect_signals()
	_switch_tab("model")

func _setup_dialogs():
	save_dialog.file_mode = FileDialog.FILE_MODE_SAVE_FILE
	load_dialog.file_mode = FileDialog.FILE_MODE_OPEN_FILE
	import_dialog.file_mode = FileDialog.FILE_MODE_OPEN_FILE
	export_dialog.file_mode = FileDialog.FILE_MODE_SAVE_FILE
	save_dialog.access = FileDialog.ACCESS_FILESYSTEM
	load_dialog.access = FileDialog.ACCESS_FILESYSTEM
	import_dialog.access = FileDialog.ACCESS_FILESYSTEM
	export_dialog.access = FileDialog.ACCESS_FILESYSTEM

func _connect_signals():
	# 路径已更新为新的 UILayer
	$UILayer/TopBar/SaveBtn.pressed.connect(_on_save_btn_pressed)
	$UILayer/TopBar/LoadBtn.pressed.connect(_on_load_btn_pressed)
	$UILayer/TopBar/ImportBtn.pressed.connect(_on_import_btn_pressed)
	$UILayer/TopBar/ExportBtn.pressed.connect(_on_export_btn_pressed)
	
	$UILayer/ButtomBar/model.pressed.connect(func(): _switch_tab("model"))
	$UILayer/ButtomBar/train.pressed.connect(func(): _switch_tab("train"))
	$UILayer/ButtomBar/test.pressed.connect(func(): _switch_tab("test"))
	
	save_dialog.file_selected.connect(_on_save_file)
	load_dialog.file_selected.connect(_on_load_file)
	import_dialog.file_selected.connect(_on_import_file)
	export_dialog.file_selected.connect(_on_export_file)
	$Dialogs/ExportWindow/VBoxContainer/ConfirmExportBtn.pressed.connect(func(): export_window.hide(); export_dialog.popup_centered_ratio(0.6))

func _on_save_btn_pressed():
	_flush_editor_state()
	save_dialog.clear_filters()
	save_dialog.add_filter(tab_extensions[current_tab].filter, tab_extensions[current_tab].desc)
	save_dialog.popup_centered_ratio(0.6)

func _on_load_btn_pressed():
	load_dialog.clear_filters()
	load_dialog.add_filter(tab_extensions[current_tab].filter, tab_extensions[current_tab].desc)
	load_dialog.popup_centered_ratio(0.6)

func _on_import_btn_pressed():
	import_dialog.clear_filters()
	import_dialog.add_filter("*.bpnn", "Model Blueprint (导入为节点)")
	import_dialog.popup_centered_ratio(0.6)

func _on_export_btn_pressed():
	_flush_editor_state()
	export_dialog.clear_filters()
	export_dialog.add_filter("*.py", "Python Script")
	export_window.popup_centered()

func _switch_tab(tab_name: String):
	_flush_editor_state() 
	
	# 【核心修复】：切换前，把当前的图存入专属工作区；然后取出新工作区的图
	workspaces[current_tab] = GlobalData.project_graphs.duplicate(true)
	current_tab = tab_name
	GlobalData.project_graphs = workspaces[current_tab].duplicate(true)
	GlobalData.current_path = ["main"]
	
	if current_editor: current_editor.queue_free()
	var scene_path = "res://main/" + tab_name + "_editor.tscn"
		
	if ResourceLoader.exists(scene_path):
		current_editor = load(scene_path).instantiate()
		editor_container.add_child(current_editor)

func _flush_editor_state():
	if current_editor and current_editor.has_method("save_current_graph_state"):
		current_editor.save_current_graph_state()

func _process(_delta):
	GlobalData.socket.poll()
	if GlobalData.socket.get_ready_state() == WebSocketPeer.STATE_OPEN:
		while GlobalData.socket.get_available_packet_count() > 0:
			var packet = GlobalData.socket.get_packet()
			var json = JSON.parse_string(packet.get_string_from_utf8())
			if typeof(json) == TYPE_DICTIONARY:
				if json.has("type") and json["type"] == "SHAPE_RESULTS":
					if current_editor and current_editor.has_method("apply_shape_results"):
						current_editor.apply_shape_results(json["data"])
				elif json.has("type") and json["type"] == "EXPORT_SUCCESS":
					_show_message("✅ 导出成功！", json["msg"])
				elif json.has("type") and json["type"] == "EXPORT_ERROR":
					_show_message("❌ 导出失败！", json["msg"])

func _on_save_file(path: String):
	var real_path = ProjectSettings.globalize_path(path)
	var ext = tab_extensions[current_tab].ext
	if not real_path.ends_with(ext): real_path += ext
	var file = FileAccess.open(real_path, FileAccess.WRITE)
	if file:
		file.store_string(JSON.stringify(GlobalData.project_graphs, "\t"))
		_show_message("✅ 保存成功！", "已保存到：\n" + real_path)

func _on_load_file(path: String):
	var real_path = ProjectSettings.globalize_path(path)
	var file = FileAccess.open(real_path, FileAccess.READ)
	if file:
		var json = JSON.parse_string(file.get_as_text())
		if typeof(json) == TYPE_DICTIONARY:
			GlobalData.project_graphs = json
			GlobalData.current_path = ["main"]
			if current_editor and current_editor.has_method("load_graph_state"):
				current_editor.load_graph_state("main")

func _on_import_file(path: String):
	var real_path = ProjectSettings.globalize_path(path)
	var file = FileAccess.open(real_path, FileAccess.READ)
	if file:
		var json = JSON.parse_string(file.get_as_text())
		if typeof(json) == TYPE_DICTIONARY:
			var model_name = path.get_file().get_basename()
			if current_editor and current_editor.has_method("import_model_as_group"):
				current_editor.import_model_as_group(json, model_name)
			else:
				_show_message("⚠️ 无法导入", "只有【训练/测试】界面支持导入网络！")

func _on_export_file(path: String):
	var real_path = ProjectSettings.globalize_path(path)
	if not real_path.ends_with(".py"): real_path += ".py"
	var req = {"path": real_path, "class_name": export_class_input.text, "data": GlobalData.project_graphs}
	
	if current_tab == "model": GlobalData.socket.send_text("EXPORT_PY:" + JSON.stringify(req))
	elif current_tab == "train": GlobalData.socket.send_text("EXPORT_TRAIN_PY:" + JSON.stringify(req))
	elif current_tab == "test": GlobalData.socket.send_text("EXPORT_TEST_PY:" + JSON.stringify(req)) # 【新增】

func _show_message(title_text: String, msg: String):
	message_dialog.title = title_text; message_dialog.dialog_text = msg
	message_dialog.popup_centered()
