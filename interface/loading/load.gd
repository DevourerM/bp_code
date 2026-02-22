extends Control

@onready var label = $Label
@onready var progress_ring = $ProgressRing

var is_loading = false
var connection_timer: float = 0.0
var has_requested = false

func _ready():
	label.text = "准备连接 Python 引擎..."
	# 初始进度
	if progress_ring: progress_ring.progress = 0.1
	
	GlobalData.socket.close()

	GlobalData.socket.inbound_buffer_size = 8388608
	GlobalData.socket.outbound_buffer_size = 8388608
	var err = GlobalData.socket.connect_to_url("ws://127.0.0.1:8765")
	is_loading = (err == OK)

func _process(delta):
	if not is_loading: return
	GlobalData.socket.poll()
	var state = GlobalData.socket.get_ready_state()
	
	match state:
		WebSocketPeer.STATE_CONNECTING:
			connection_timer += delta
			label.text = "正在寻找 Python 服务器 (%.1fs)..." % connection_timer
		WebSocketPeer.STATE_OPEN:
			if not has_requested:
				GlobalData.socket.send_text("REQUEST_NODES")
				has_requested = true
				label.text = "连接成功！请求神经网络数据..."
			_listen_for_lib()
		WebSocketPeer.STATE_CLOSED:
			label.text = "连接失败，请确认后端已启动"

func _listen_for_lib():
	while GlobalData.socket.get_available_packet_count() > 0:
		var packet = GlobalData.socket.get_packet()
		var json = JSON.parse_string(packet.get_string_from_utf8())
		if json and json.get("type") == "LIBRARY":
			_parse_with_animation(json.get("data", {}))

func _parse_with_animation(lib_data):
	is_loading = false
	var total = 0
	for cat in lib_data.values(): total += cat.size()
	
	var count = 0
	for cat_name in lib_data.keys():
		for node_name in lib_data[cat_name].keys():
			count += 1
			label.text = "解析节点: %s" % node_name
			if progress_ring: progress_ring.progress = 0.4 + (0.6 * count / total)
			await get_tree().process_frame # 必须等待一帧以显示进度
			
	GlobalData.node_library = lib_data
	GlobalData.library_updated.emit() # 发射库加载完成信号
	GlobalData.socket_connected.emit() # 发射连接就绪信号
	
	label.text = "加载完成！"
	await get_tree().create_timer(0.5).timeout
	get_tree().change_scene_to_file("res://main/app.tscn")
