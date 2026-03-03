extends Node

var backend_pid: int = -1

func _ready():
	# 拦截 Godot 的默认关闭行为，让我们有机会先杀掉后台进程
	get_tree().set_auto_accept_quit(false)
	_start_python_backend()

func _start_python_backend():
	if OS.has_feature("editor"):
		print("当前处于编辑器开发模式：请手动在 VSCode 运行 Python 后端。")
		return 
		
	var base_dir = OS.get_executable_path().get_base_dir()
	
	# 1. 找到便携版 Python 的路径
	var python_exe = base_dir.path_join("python_env/python.exe")
	# 2. 找到你要运行的脚本路径
	var script_path = base_dir.path_join("backend_scripts/server.py")
	
	if FileAccess.file_exists(python_exe) and FileAccess.file_exists(script_path):
		# 让 Python.exe 运行对应的脚本
		backend_pid = OS.create_process(python_exe, [script_path])
		print("🚀 内部绿色版 Python 引擎已启动，PID:", backend_pid)
	else:
		print("❌ 严重错误：未找到便携版 Python 或后端脚本！")

# 监听软件关闭事件
func _notification(what):
	if what == NOTIFICATION_WM_CLOSE_REQUEST:
		# 软件关闭前，杀掉 Python 后台进程
		if backend_pid != -1:
			OS.kill(backend_pid)
			print("🛑 Python 引擎已安全关闭。")
		get_tree().quit() # 最终退出 Godot
