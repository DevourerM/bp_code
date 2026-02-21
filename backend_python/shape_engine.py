import torch
import torch.nn as nn
import ast
import json

def parse_val(val_str, expected_type):
    if val_str == "" or val_str == "None": return None
    try:
        if expected_type == "tuple": return ast.literal_eval(val_str)
        elif expected_type == "int": return int(float(val_str))
        elif expected_type == "float": return float(val_str)
        elif expected_type == "bool": return str(val_str).lower() == "true"
    except: pass
    try: return ast.literal_eval(val_str)
    except: return val_str

def infer_graph_shapes(project_data):
    global_shapes = {}
    global_updates = {}
    _TENSOR_CACHE = {}
    visited_graphs = set() # 【新增】：记录编译过的图，防止死循环

    def infer_subgraph(graph_id, parent_inputs=None):
        if graph_id not in project_data: return None
        visited_graphs.add(graph_id)
            
        graph = project_data[graph_id]
        nodes = graph.get("nodes", {})
        connections = graph.get("connections", [])
        
        tensor_cache = {}
        incoming_tensors = {nid: {} for nid in nodes.keys()}
        required_inputs = {}
        
        data_input_nodes = [nid for nid, info in nodes.items() if info["type"] == "Data Input"]
        
        def sort_key(nid):
            nid_str = str(nid)
            if nid_str.startswith("input") and nid_str[5:].isdigit():
                return int(nid_str[5:])
            return nodes[nid].get("pos_y", 0)
            
        data_input_nodes.sort(key=sort_key)
        
        for i, nid in enumerate(data_input_nodes):
            # 【修复核心】：如果外层馈送了数据，优先使用外层数据；否则使用子图节点的局部默认值
            if parent_inputs is not None and i < len(parent_inputs) and parent_inputs[i] is not None and not isinstance(parent_inputs[i], str):
                tensor_cache[nid] = parent_inputs[i]
            else:
                p = nodes[nid].get("params", {})
                sh_val = p.get("shape", {}).get("value", "(1, 3, 224, 224)")
                md_val = p.get("mode", {}).get("value", "randn")
                sh = parse_val(sh_val, "tuple")
                try:
                    if md_val == "ones": tensor_cache[nid] = torch.ones(*sh)
                    elif md_val == "zeros": tensor_cache[nid] = torch.zeros(*sh)
                    else: tensor_cache[nid] = torch.randn(*sh)
                except:
                    tensor_cache[nid] = "输入错误"
        
        for nid, info in nodes.items():
            ins = info.get("inputs", [])
            required_inputs[nid] = len(ins) if ins else (1 if info.get("main_in") else 0)
            
        pending_conns = connections.copy()
        processed_count = -1
        
        for _ in range(100):
            if len(pending_conns) == processed_count: break
            processed_count = len(pending_conns)
            remaining_conns = []
            
            for conn in pending_conns:
                f, t, tp = conn["from"], conn["to"], conn.get("to_port", 0)
                if f not in tensor_cache:
                    remaining_conns.append(conn)
                    continue
                    
                in_t = tensor_cache[f]
                conn_key = f"{f}->{t}"
                global_shapes[conn_key] = str(list(in_t.shape)) if not isinstance(in_t, str) else in_t
                incoming_tensors[t][tp] = in_t
                
            for nid, info in nodes.items():
                if nid in tensor_cache: continue
                req = required_inputs[nid]
                has_t = incoming_tensors[nid]
                
                # 【修复核心】：允许 Group 节点在没有插满线时部分执行！
                is_ready = False
                if req == 0: is_ready = True
                elif len(has_t) == req: is_ready = True
                elif info["type"] in ["Group", "Loop"]: is_ready = True
                
                if is_ready:
                    l_type = info["type"]
                    p_raw = info.get("params", {})
                    p = {k: parse_val(v.get("value", ""), v.get("type", "string")) for k,v in p_raw.items()}
                    
                    try:
                        overridden = {}
                        main_t = has_t.get(0)
                        
                        if l_type == "Group":
                            sub_inputs = [has_t.get(i) for i in range(req)]
                            sub_out = infer_subgraph(nid, parent_inputs=sub_inputs)
                            out_t = sub_out if sub_out is not None else (main_t if main_t is not None else "等待...")
                            
                        elif l_type == "Loop":
                            iters = p.get("iterations", 3)
                            curr_t = main_t
                            for step in range(iters):
                                sub_out = infer_subgraph(nid, parent_inputs=[curr_t])
                                if sub_out is not None: curr_t = sub_out
                            out_t = curr_t
                            
                        elif l_type == "Value Display":
                            idx = p.get("index", (0,))
                            try: overridden["result"] = str(round(main_t[idx].item(), 6))
                            except: overridden["result"] = "越界错误"
                            out_t = main_t
                            
                        elif hasattr(nn, l_type):
                            auto_k = "in_channels" if "Conv" in l_type else ("num_features" if "BatchNorm" in l_type else ("in_features" if "Linear" in l_type else None))
                            if auto_k and len(main_t.shape) >= (2 if "Conv" in l_type else 1):
                                dim_idx = 1 if "Conv" in l_type or "BatchNorm" in l_type else -1
                                p[auto_k] = main_t.shape[dim_idx]
                                overridden[auto_k] = str(p[auto_k])
                            
                            layer = getattr(nn, l_type)(**p)
                            out_t = layer(main_t)
                            
                        elif l_type == "Concat":
                            out_t = torch.cat((main_t, has_t.get(1)), dim=p.get("dim", 1))
                        elif l_type == "Math":
                            op = p.get("op", "add")
                            if "add" in op: out_t = main_t + has_t.get(1)
                            elif "sub" in op: out_t = main_t - has_t.get(1)
                            elif "mul" in op: out_t = main_t * has_t.get(1)
                            elif "matmul" in op: out_t = torch.matmul(main_t, has_t.get(1))
                        else:
                            out_t = main_t
                            
                        if overridden: global_updates[nid] = overridden
                        tensor_cache[nid] = out_t
                        
                    except Exception as e:
                        tensor_cache[nid] = "等待参数..."
                        
            pending_conns = remaining_conns
            
        data_out_nodes = [nid for nid, info in nodes.items() if info["type"] == "Data Output"]
        if data_out_nodes and data_out_nodes[0] in tensor_cache:
            return tensor_cache[data_out_nodes[0]]
        return None

    # 1. 优先编译主空间
    infer_subgraph("main")
    
    # 2. 【修复核心】：强制编译所有没被 main 激活的子空间，确保你在里层连线时能独立查看结果！
    for gid in list(project_data.keys()):
        if gid not in visited_graphs:
            infer_subgraph(gid)
            
    return {"shapes": global_shapes, "updated_params": global_updates}