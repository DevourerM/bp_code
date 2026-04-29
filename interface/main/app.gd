extends Control

# ==========================================
# 1. UI 组件引用
# ==========================================
@onready var editor_container = $EditorContainer
@onready var save_dialog: FileDialog = $Dialogs/SaveDialog
@onready var load_dialog: FileDialog = $Dialogs/LoadDialog
@onready var import_dialog: FileDialog = $Dialogs/ImportDialog
@onready var export_dialog: FileDialog = $Dialogs/ExportDialog
@onready var export_window: Window = $Dialogs/ExportWindow
@onready var export_class_input: LineEdit = $Dialogs/ExportWindow/VBoxContainer/ClassInput
@onready var message_dialog: AcceptDialog = $Dialogs/MessageDialog

# ==========================================
# 2. 全局状态变量
# ==========================================
var current_editor: Node = null
var current_tab: String = "model" 
var python_pid: int = -1 # 记录 Python 后端进程 PID，以便退出时精准击杀

# 独立工作区缓存，保证切换页面时互不干扰
var workspaces = {
	"model": {}, "train": {}, "test": {}
}

# 各界面对应的专属文件后缀与过滤说明
var tab_extensions = {
	"model": { "filter": "*.bpnn", "ext": ".bpnn", "desc": "Model Blueprint" },
	"train": { "filter": "*.bptr", "ext": ".bptr", "desc": "Train Blueprint" },
	"test":  { "filter": "*.bpte", "ext": ".bpte", "desc": "Test Blueprint" }
}

# ==========================================
# 3. 生命周期与初始化
# ==========================================
func _ready():
	_start_python_backend() # 优先启动便携式 Python 环境
	_setup_dialogs()
	_connect_signals()
	_switch_tab("model")

func _process(_delta):
	# 轮询 WebSocket 消息
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

func _notification(what):
	# 核心清理机制：关闭软件时，杀掉 Python 进程！防止产生僵尸进程
	if what == NOTIFICATION_WM_CLOSE_REQUEST:
		if python_pid != -1:
			print("正在清理后台 Python 进程 (PID: ", python_pid, ")...")
			OS.kill(python_pid)

# ==========================================
# 4. 后端 Python 进程管理
# ==========================================
func _start_python_backend():
	# 【核心修改】：如果是开发调试环境，直接拦截跳过！
	if OS.has_feature("editor"):
		print("🔧 [开发模式] 已跳过 Python 自动启动。请确保您已手动运行了 server.py！")
		return
		
	# --- 以下只有在导出的 .exe 正式环境中才会执行 ---
	
	# 获取当前运行的 .exe 文件所在的绝对目录
	var base_dir = OS.get_executable_path().get_base_dir()
	
	# 设置打包后的相对路径
	var python_exe = base_dir.path_join("python_env/python.exe")
	var server_script = base_dir.path_join("backend_scripts/server.py")
	
	# 智能嗅探：优先使用 pythonw.exe 隐藏控制台黑框
	if FileAccess.file_exists(base_dir.path_join("python_env/pythonw.exe")):
		python_exe = base_dir.path_join("python_env/pythonw.exe")
		
	print("🚀 [生产模式] 尝试启动 Python 后端: ", python_exe, " -> ", server_script)
	
	# 安全检查（防止用户误删文件）
	if not FileAccess.file_exists(python_exe):
		push_error("致命错误：找不到自带的 Python 环境！请检查 python_env 文件夹。")
		# 可选：如果找不到环境，可以弹窗提醒用户
		_show_message("环境缺失", "找不到 Python 运行环境，请确保 python_env 文件夹与 exe 放在一起！")
		return
		
	# 异步启动程序，绝对不会阻塞 Godot 主线程
	python_pid = OS.create_process(python_exe, [server_script], false)
	print("✅ Python 后端已成功启动，PID: ", python_pid)

# ==========================================
# 5. UI 构建与信号连接
# ==========================================
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
	# 顶部操作栏
	$UILayer/TopBar/SaveBtn.pressed.connect(_on_save_btn_pressed)
	$UILayer/TopBar/LoadBtn.pressed.connect(_on_load_btn_pressed)
	$UILayer/TopBar/ImportBtn.pressed.connect(_on_import_btn_pressed)
	$UILayer/TopBar/ExportBtn.pressed.connect(_on_export_btn_pressed)
	
	# 底部 Tab 切换
	$UILayer/ButtomBar/model.pressed.connect(func(): _switch_tab("model"))
	$UILayer/ButtomBar/train.pressed.connect(func(): _switch_tab("train"))
	$UILayer/ButtomBar/test.pressed.connect(func(): _switch_tab("test"))
	
	# 文件对话框事件
	save_dialog.file_selected.connect(_on_save_file)
	load_dialog.file_selected.connect(_on_load_file)
	import_dialog.file_selected.connect(_on_import_file)
	export_dialog.file_selected.connect(_on_export_file)
	
	$Dialogs/ExportWindow/VBoxContainer/ConfirmExportBtn.pressed.connect(func(): 
		export_window.hide() 
		export_dialog.popup_centered_ratio(0.6)
	)

# ==========================================
# 6. 工作区调度逻辑
# ==========================================
func _switch_tab(tab_name: String):
	_flush_editor_state() 
	
	# 工作区状态隔离：将当前的图存入缓存，抽出新工作区的图
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
	# 触发子编辑器保存当前状态至 GlobalData
	if current_editor and current_editor.has_method("save_current_graph_state"):
		current_editor.save_current_graph_state()

# ==========================================
# 7. 文件 I/O 与后端通讯
# ==========================================
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
	
	var req = {
		"path": real_path, 
		"class_name": export_class_input.text, 
		"data": GlobalData.project_graphs
	}
	
	# 根据当前所处界面，向 Python 后端发送不同的编译指令
	if current_tab == "model": GlobalData.socket.send_text("EXPORT_PY:" + JSON.stringify(req))
	elif current_tab == "train": GlobalData.socket.send_text("EXPORT_TRAIN_PY:" + JSON.stringify(req))
	elif current_tab == "test": GlobalData.socket.send_text("EXPORT_TEST_PY:" + JSON.stringify(req)) 

func _show_message(title_text: String, msg: String):
	message_dialog.title = title_text
	message_dialog.dialog_text = msg
	message_dialog.popup_centered()
