extends Node

signal library_updated
signal socket_connected
signal shapes_updated(data)

var socket = WebSocketPeer.new()
var node_library = {}

# ==========================================
# 【新增】：多层级图结构管理器
# ==========================================
# 存储所有子空间的图数据结构：{"main": {nodes:{...}, conns:[]}, "Group_1": {...}}
var project_graphs = {
	"main": {"nodes": {}, "connections": []}
}
# 当前所在的面包屑路径，例如 ["main", "Group_1"]
var current_path = ["main"] 

func get_current_graph_id() -> String:
	return current_path.back()
