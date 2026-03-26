import torch
import torch.nn as nn
import ast
import traceback

# ==========================================
# 核心引擎：实时张量形状与维度推导 
# ==========================================
def infer_graph_shapes(project_data):
    main_graph = project_data.get("main", {"nodes": {}, "connections": []})
    nodes = main_graph.get("nodes", {})
    connections = main_graph.get("connections", [])

    # 1. 拓扑排序 (保证推导顺序从前到后)
    in_degree = {nid: 0 for nid in nodes}
    adj_list = {nid: [] for nid in nodes}
    for conn in connections:
        from_n = conn["from"]
        to_n = conn["to"]
        if from_n in nodes and to_n in nodes:
            adj_list[from_n].append(conn)
            in_degree[to_n] += 1

    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    topo_order = []
    while queue:
        curr = queue.pop(0)
        topo_order.append(curr)
        for conn in adj_list[curr]:
            to_n = conn["to"]
            in_degree[to_n] -= 1
            if in_degree[to_n] == 0:
                queue.append(to_n)

    out_vars = {}        # 记录每个节点的输出 Tensor
    shape_results = {}   # 记录前端需要展示的形状字符串
    updated_params = {}  # 记录需要自动填入前端的参数字典

    # 2. 依次执行节点推导
    for nid in topo_order:
        info = nodes[nid]
        ntype = info.get("type", "")
        params = info.get("params", {})

        # 【独立处理】：配置连线的专属 UI 标签
        if ntype in ["Weight Init", "Comment", "Inference Config", "Training Config", "Dataset Loader", "Target Loader"]:
            for conn in connections:
                if conn["from"] == nid:
                    conn_key = f"{conn['from']}->{conn['to']}"
                    if ntype == "Weight Init":
                        shape_results[conn_key] = "⚙️ 初始化"
                    else:
                        shape_results[conn_key] = "⚙️ 配置线"
            continue

        # 收集当前计算层的所有有效输入 Tensor
        in_ports = {}
        for conn in connections:
            if conn["to"] == nid:
                from_type = nodes.get(conn["from"], {}).get("type", "")
                if from_type in ["Weight Init", "Comment"]:
                    continue # 物理隔离配置线
                    
                if conn["from"] in out_vars and conn["from_port"] in out_vars[conn["from"]]:
                    in_ports[conn["to_port"]] = out_vars[conn["from"]][conn["from_port"]]

        in_args = [in_ports[i] for i in sorted(in_ports.keys())]

        try:
            # ---------------------------------------------
            # 模拟执行与形状推演
            # ---------------------------------------------
            if ntype == "Data Input":
                shape_str = params.get("shape", {}).get("value", "(1, 3, 224, 224)")
                shape_tuple = ast.literal_eval(shape_str) if isinstance(shape_str, str) else shape_str
                out_vars[nid] = {0: torch.zeros(shape_tuple)}

            elif ntype == "Data Output" or ntype == "Group":
                if in_args: out_vars[nid] = {0: in_args[0]}
            
            # --- 张量形态类 ---
            elif ntype == "Reshape":
                shape_str = params.get("shape", {}).get("value", "(1, -1)")
                s = ast.literal_eval(shape_str) if isinstance(shape_str, str) else shape_str
                out_vars[nid] = {0: in_args[0].view(s)}
            elif ntype == "Permute":
                dims_str = params.get("dims", {}).get("value", "(0, 2, 1, 3)")
                d = ast.literal_eval(dims_str) if isinstance(dims_str, str) else dims_str
                out_vars[nid] = {0: in_args[0].permute(d)}
            elif ntype == "Squeeze":
                dim_str = str(params.get("dim", {}).get("value", "None"))
                if dim_str != "None" and dim_str != "": out_vars[nid] = {0: in_args[0].squeeze(int(dim_str))}
                else: out_vars[nid] = {0: in_args[0].squeeze()}
            elif ntype == "Unsqueeze":
                dim = int(float(params.get("dim", {}).get("value", "1")))
                out_vars[nid] = {0: in_args[0].unsqueeze(dim)}
            elif ntype == "Expand":
                sizes_str = params.get("sizes", {}).get("value", "(1, 8, 8)")
                s = ast.literal_eval(sizes_str) if isinstance(sizes_str, str) else sizes_str
                out_vars[nid] = {0: in_args[0].expand(s)}

            # --- 数学运算类 ---
            elif ntype == "Binary Math":
                op = params.get("op", {}).get("value", "matmul (@)")
                if not in_args or len(in_args) < 2: continue 
                a, b = in_args[0], in_args[1]
                if "add" in op: out_vars[nid] = {0: a + b}
                elif "sub" in op: out_vars[nid] = {0: a - b}
                elif "mul" in op: out_vars[nid] = {0: a * b}
                elif "div" in op: out_vars[nid] = {0: a / b}
                elif "matmul" in op: out_vars[nid] = {0: torch.matmul(a, b)}
            elif ntype == "Concat":
                dim = int(float(params.get("dim", {}).get("value", "-1")))
                out_vars[nid] = {0: torch.cat(in_args, dim=dim)}
            
            # --- PyTorch 原生网络层 ---
            elif hasattr(nn, ntype):
                kwargs = {}
                for k, v in params.items():
                    val = v.get("value", v.get("default", ""))
                    if val == "" or str(val) == "None": continue
                    
                    # 强转所有冒充 float 的 int 参数
                    if isinstance(val, float) and val.is_integer():
                        val = int(val) 
                    elif isinstance(val, str):
                        try:
                            parsed_val = ast.literal_eval(val)
                            if isinstance(parsed_val, float) and parsed_val.is_integer():
                                val = int(parsed_val)
                            else:
                                val = parsed_val
                        except Exception:
                            pass 
                            
                    kwargs[k] = val
                
                # 自动嗅探与智能填充 In_Channels 等
                if in_args:
                    in_tensor_shape = in_args[0].shape
                    
                    if "Conv" in ntype and len(in_tensor_shape) >= 2:
                        auto_in_channels = in_tensor_shape[1]
                        kwargs["in_channels"] = auto_in_channels
                        if nid not in updated_params: updated_params[nid] = {}
                        updated_params[nid]["in_channels"] = str(auto_in_channels)
                        
                    elif "Linear" in ntype and len(in_tensor_shape) >= 1:
                        auto_in_features = in_tensor_shape[-1]
                        kwargs["in_features"] = auto_in_features
                        if nid not in updated_params: updated_params[nid] = {}
                        updated_params[nid]["in_features"] = str(auto_in_features)
                        
                    elif "BatchNorm" in ntype and len(in_tensor_shape) >= 2:
                        auto_num_features = in_tensor_shape[1]
                        kwargs["num_features"] = auto_num_features
                        if nid not in updated_params: updated_params[nid] = {}
                        updated_params[nid]["num_features"] = str(auto_num_features)
                            
                # 实例化网络层
                layer = getattr(nn, ntype)(**kwargs)
                out = layer(*in_args)
                out_vars[nid] = {0: out}

            # 成功！给当前节点的【输出线】标上正确的形状
            for conn in connections:
                if conn["from"] == nid:
                    port = conn["from_port"]
                    if port in out_vars[nid]:
                        tensor = out_vars[nid][port]
                        conn_key = f"{conn['from']}->{conn['to']}"
                        shape_results[conn_key] = str(tuple(tensor.shape))

        # ==========================================
        # 【核心逻辑修正】：报错只影响从当前节点连出去的线！
        # ==========================================
        except TypeError as e:
            err_str = str(e)
            for conn in connections:
                # 【修改点】：不再是 conn["to"]，而是只找连出去的线 conn["from"]
                if conn["from"] == nid:
                    conn_key = f"{conn['from']}->{conn['to']}"
                    if "missing" in err_str and "required" in err_str:
                        shape_results[conn_key] = "⏳ 等待输入必填参数..."
                    else:
                        shape_results[conn_key] = f"❌ 参数错误: 请检查输入"
            
            if not ("missing" in err_str and "required" in err_str):
                print(f"节点 {nid} ({ntype}) 推导失败: {e}")

        except Exception as e:
            for conn in connections:
                # 同理：报错只污染下家，绝不反咬上家！
                if conn["from"] == nid:
                    conn_key = f"{conn['from']}->{conn['to']}"
                    shape_results[conn_key] = f"❌ 维度冲突: {type(e).__name__}"
            print(f"节点 {nid} ({ntype}) 推导失败: {e}")

    return {
        "shapes": shape_results,
        "updated_params": updated_params
    }