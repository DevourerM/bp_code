import torch
import torch.nn as nn
import ast
import traceback

# ==========================================
# 核心引擎：实时推导 (V4.1 多端口动态透传版)
# ==========================================
def infer_graph_shapes(project_data):
    main_graph = project_data.get("main", {"nodes": {}, "connections": []})
    nodes = main_graph.get("nodes", {})
    connections = main_graph.get("connections", [])

    in_degree = {nid: 0 for nid in nodes}
    adj_list = {nid: [] for nid in nodes}
    
    # 构建基础邻接表
    for conn in connections:
        if conn["from"] in nodes and conn["to"] in nodes:
            adj_list[conn["from"]].append(conn)
            in_degree[conn["to"]] += 1

    # 拓扑排序：优先推导没有依赖的输入节点
    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    topo_order = []
    while queue:
        curr = queue.pop(0)
        topo_order.append(curr)
        for conn in adj_list[curr]:
            in_degree[conn["to"]] -= 1
            if in_degree[conn["to"]] == 0: queue.append(conn["to"])

    out_vars = {nid: {} for nid in nodes}        
    shape_results = {}   
    updated_params = {}  

    # 分离出两次遍历：第一次计算全部普通节点，第二次专门处理 Call Function (因为它需要依赖 Return 的结果)
    for pass_num in [1, 2]:
        for nid in topo_order:
            info = nodes[nid]
            ntype = info.get("type", "")
            params = info.get("params", {})

            if pass_num == 1 and ntype == "Call Function": continue
            if pass_num == 2 and ntype != "Call Function": continue

            if ntype in ["Weight Init", "Comment", "Inference Config", "Training Config", "Dataset Loader", "Target Loader", "Def Function"]:
                for conn in connections:
                    if conn["from"] == nid:
                        if ntype == "Weight Init": shape_results[f"{conn['from']}->{conn['to']}"] = "⚙️ 初始化"
                        elif ntype == "Def Function": shape_results[f"{conn['from']}->{conn['to']}"] = "➡️ 函数绑定"
                        else: shape_results[f"{conn['from']}->{conn['to']}"] = "⚙️ 配置线"
                continue

            in_ports = {}
            for conn in connections:
                if conn["to"] == nid:
                    if nodes.get(conn["from"], {}).get("type", "") in ["Weight Init", "Comment", "Def Function"]: continue
                    if conn["from"] in out_vars and conn["from_port"] in out_vars[conn["from"]]:
                        in_ports[conn["to_port"]] = out_vars[conn["from"]][conn["from_port"]]

            in_args = [in_ports[i] for i in sorted(in_ports.keys())]

            try:
                if ntype == "Data Input":
                    shape_str = params.get("shape", {}).get("value", "(1, 3, 224, 224)")
                    shape_tuple = tuple(int(x) for x in ast.literal_eval(shape_str))
                    out_vars[nid] = {0: torch.zeros(shape_tuple)}

                elif ntype in ["Return Function", "Loop Begin", "Loop End"]:
                    # Return Function 如果有多个输入，将全部透传保存
                    for i in range(len(in_args)):
                        if in_args[i] is not None:
                            out_vars[nid][i] = in_args[i]
                    
                elif ntype == "Call Function":
                    # 动态读取并继承目标 Return 节点的张量尺寸，实现完美免推导！
                    fname = params.get("func_name", {}).get("value", "main")
                    out_cnt = int(float(params.get("output_count", {}).get("value", 1)))
                    
                    # 寻找目标函数的 Return 节点
                    target_return_nid = None
                    for r_nid, r_info in nodes.items():
                        if r_info.get("type") == "Return Function" and r_info.get("params", {}).get("func_name", {}).get("value", "main") == fname:
                            target_return_nid = r_nid; break
                    
                    for i in range(out_cnt):
                        if target_return_nid and i in out_vars.get(target_return_nid, {}):
                            out_vars[nid][i] = torch.zeros(out_vars[target_return_nid][i].shape)
                        else:
                            out_vars[nid][i] = torch.zeros((1, 10)) # 找不到就给个兜底
                
                # --- 张量形态类 ---
                elif ntype == "Reshape":
                    s = tuple(int(x) for x in ast.literal_eval(params.get("shape", {}).get("value", "(1, -1)")))
                    out_vars[nid] = {0: in_args[0].view(s)}
                elif ntype == "Permute":
                    d = tuple(int(x) for x in ast.literal_eval(params.get("dims", {}).get("value", "(0, 2, 1, 3)")))
                    out_vars[nid] = {0: in_args[0].permute(d)}
                elif ntype == "Squeeze":
                    d_str = str(params.get("dim", {}).get("value", "None"))
                    out_vars[nid] = {0: in_args[0].squeeze(int(float(d_str)))} if d_str != "None" and d_str != "" else {0: in_args[0].squeeze()}
                elif ntype == "Unsqueeze":
                    out_vars[nid] = {0: in_args[0].unsqueeze(int(float(params.get("dim", {}).get("value", "1"))))}
                elif ntype == "Expand":
                    s = tuple(int(x) for x in ast.literal_eval(params.get("sizes", {}).get("value", "(1, 8, 8)")))
                    out_vars[nid] = {0: in_args[0].expand(s)}
                elif ntype == "Concat":
                    out_vars[nid] = {0: torch.cat(in_args, dim=int(float(params.get("dim", {}).get("value", "-1"))))}
                elif ntype == "Binary Math":
                    op = params.get("op", {}).get("value", "add")
                    if not in_args or len(in_args) < 2: raise TypeError("missing arguments")
                    a, b = in_args[0], in_args[1]
                    if "add" in op: out_vars[nid] = {0: a + b}
                    elif "sub" in op: out_vars[nid] = {0: a - b}
                    elif "mul" in op: out_vars[nid] = {0: a * b}
                    elif "matmul" in op: out_vars[nid] = {0: torch.matmul(a, b)}
                
                # --- PyTorch 原生网络层 ---
                elif hasattr(nn, ntype):
                    kwargs = {}
                    for k, v in params.items():
                        val = v.get("value", v.get("default", ""))
                        if val == "" or str(val) == "None": continue
                        if isinstance(val, float) and val.is_integer(): val = int(val) 
                        elif isinstance(val, str):
                            try:
                                pv = ast.literal_eval(val)
                                val = int(pv) if isinstance(pv, float) and pv.is_integer() else pv
                            except Exception: pass 
                        kwargs[k] = val
                    
                    if in_args:
                        sh = in_args[0].shape
                        if "Conv" in ntype and len(sh) >= 2:
                            kwargs["in_channels"] = sh[1]
                            if nid not in updated_params: updated_params[nid] = {}
                            updated_params[nid]["in_channels"] = str(sh[1])
                        elif "Linear" in ntype and len(sh) >= 1:
                            kwargs["in_features"] = sh[-1]
                            if nid not in updated_params: updated_params[nid] = {}
                            updated_params[nid]["in_features"] = str(sh[-1])
                        elif "BatchNorm" in ntype and len(sh) >= 2:
                            kwargs["num_features"] = sh[1]
                            if nid not in updated_params: updated_params[nid] = {}
                            updated_params[nid]["num_features"] = str(sh[1])
                                
                    layer = getattr(nn, ntype)(**kwargs)
                    out_vars[nid] = {0: layer(*in_args)}

                for conn in connections:
                    if conn["from"] == nid and conn["from_port"] in out_vars.get(nid, {}):
                        shape_results[f"{conn['from']}->{conn['to']}"] = str(tuple(out_vars[nid][conn["from_port"]].shape))

            except TypeError as e:
                err_str = str(e)
                for conn in connections:
                    if conn["from"] == nid:
                        shape_results[f"{conn['from']}->{conn['to']}"] = "⏳ 等待输入..." if "missing" in err_str else "❌ 参数错误"
            except Exception as e:
                for conn in connections:
                    if conn["from"] == nid:
                        shape_results[f"{conn['from']}->{conn['to']}"] = f"❌ 维度冲突"

    for conn in connections:
        conn_key = f"{conn['from']}->{conn['to']}"
        if conn_key not in shape_results: shape_results[conn_key] = "⏳ 等待上游数据..."

    return {"shapes": shape_results, "updated_params": updated_params}