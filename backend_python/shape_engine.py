import torch
import torch.nn as nn
import ast
import traceback

def get_blocks(nodes, connections):
    """弱连通分量 (WCC) 算法：自动找出画布上所有物理相连的孤岛"""
    adj = {n: [] for n in nodes}
    for c in connections:
        if c["from"] in nodes and c["to"] in nodes:
            adj[c["from"]].append(c["to"])
            adj[c["to"]].append(c["from"])
            
    blocks = {}
    visited = set()
    for nid, info in nodes.items():
        if info.get("type") == "Def Function":
            fname = info.get("params", {}).get("func_name", {}).get("value", "main")
            comp = set()
            q = [nid]
            visited.add(nid)
            while q:
                curr = q.pop(0)
                comp.add(curr)
                for nbr in adj[curr]:
                    if nbr not in visited:
                        visited.add(nbr)
                        q.append(nbr)
            blocks[fname] = comp
    return blocks

def infer_graph_shapes(project_data):
    nodes = project_data.get("main", {}).get("nodes", {})
    connections = project_data.get("main", {}).get("connections", [])

    blocks = get_blocks(nodes, connections)

    # 预计算每个函数模块内部的拓扑执行顺序
    block_topos = {}
    for fname, comp in blocks.items():
        in_deg = {n: 0 for n in comp}
        dir_adj = {n: [] for n in comp}
        for c in connections:
            fn, tn = c["from"], c["to"]
            # 切断配置线的推导污染
            if fn in comp and tn in comp and nodes.get(fn, {}).get("type") not in ["Weight Init", "Comment", "Def Function"]:
                dir_adj[fn].append(tn)
                in_deg[tn] += 1
        q = [n for n in comp if in_deg[n] == 0]
        topo = []
        while q:
            curr = q.pop(0)
            topo.append(curr)
            for nbr in dir_adj[curr]:
                in_deg[nbr] -= 1
                if in_deg[nbr] == 0: q.append(nbr)
        block_topos[fname] = topo

    shape_results = {}
    updated_params = {}

    # ==========================================
    # 【核心沙盒】：动态递归推演解释器
    # ==========================================
    def eval_block(fname, actual_inputs=None, call_stack=None):
        if call_stack is None: call_stack = set()
        if fname in call_stack: raise RecursionError(f"检测到死循环调用: {fname}")
        if fname not in blocks: raise ValueError(f"画布上未找到函数模块: {fname}")

        call_stack.add(fname)
        topo = block_topos[fname]
        comp = blocks[fname]

        # 排序入口参数以保证严格对齐
        data_inputs = [n for n in topo if nodes[n].get("type") == "Data Input"]
        data_inputs.sort(key=lambda x: nodes[x].get("params", {}).get("arg_name", {}).get("value", "x"))

        out_vars = {n: {} for n in topo}

        for nid in topo:
            info = nodes[nid]
            ntype = info.get("type", "")
            params = info.get("params", {})

            # 处理静态界面标签
            if ntype in ["Weight Init", "Comment", "Inference Config", "Training Config", "Dataset Loader", "Target Loader", "Def Function"]:
                for c in connections:
                    if c["from"] == nid:
                        shape_results[f"{c['from']}->{c['to']}"] = "⚙️ 初始化" if ntype == "Weight Init" else ("➡️ 函数绑定" if ntype == "Def Function" else "⚙️ 配置参数")
                continue

            # 收集有效输入
            in_ports = {}
            for c in connections:
                if c["to"] == nid and c["from"] in comp:
                    if c["from"] in out_vars and c["from_port"] in out_vars[c["from"]]:
                        in_ports[c["to_port"]] = out_vars[c["from"]][c["from_port"]]

            in_args = [in_ports.get(i) for i in range(max(in_ports.keys()) + 1)] if in_ports else []

            try:
                if ntype == "Data Input":
                    idx = data_inputs.index(nid)
                    # 【核心魔法】：如果 Call Function 传来了真实数据，强行覆盖掉预览尺寸！
                    if actual_inputs is not None and idx < len(actual_inputs) and actual_inputs[idx] is not None:
                        out_vars[nid] = {0: actual_inputs[idx]}
                    else:
                        s = tuple(int(x) for x in ast.literal_eval(params.get("shape", {}).get("value", "(1, 3, 224, 224)")))
                        out_vars[nid] = {0: torch.zeros(s)}

                elif ntype == "Call Function":
                    c_fname = params.get("func_name", {}).get("value", "")
                    out_cnt = int(float(params.get("output_count", {}).get("value", 1)))
                    valid_args = [x for x in in_args if x is not None]
                    
                    # 【递归触发】：带入真实数据，启动子图沙盒推演！
                    sub_out = eval_block(c_fname, valid_args, call_stack.copy())
                    
                    for i in range(out_cnt):
                        if i < len(sub_out) and sub_out[i] is not None:
                            out_vars[nid][i] = torch.zeros(sub_out[i].shape)
                        else:
                            out_vars[nid][i] = torch.zeros((1, 10)) # 兜底

                elif ntype in ["Return Function", "Loop Begin", "Loop End"]:
                    for i in range(len(in_args)):
                        if in_args[i] is not None: out_vars[nid][i] = in_args[i]

                # --- 基础形状变换 ---
                elif ntype == "Reshape": out_vars[nid] = {0: in_args[0].view(tuple(int(x) for x in ast.literal_eval(params.get("shape", {}).get("value", "(1, -1)"))))}
                elif ntype == "Permute": out_vars[nid] = {0: in_args[0].permute(tuple(int(x) for x in ast.literal_eval(params.get("dims", {}).get("value", "(0, 2, 1, 3)"))))}
                elif ntype == "Squeeze":
                    d_str = str(params.get("dim", {}).get("value", "None"))
                    out_vars[nid] = {0: in_args[0].squeeze(int(float(d_str)))} if d_str != "None" and d_str != "" else {0: in_args[0].squeeze()}
                elif ntype == "Unsqueeze": out_vars[nid] = {0: in_args[0].unsqueeze(int(float(params.get("dim", {}).get("value", "1"))))}
                elif ntype == "Expand": out_vars[nid] = {0: in_args[0].expand(tuple(int(x) for x in ast.literal_eval(params.get("sizes", {}).get("value", "(1, 8, 8)"))))}
                elif ntype == "Concat": out_vars[nid] = {0: torch.cat(in_args, dim=int(float(params.get("dim", {}).get("value", "-1"))))}
                elif ntype == "Binary Math":
                    op = params.get("op", {}).get("value", "add")
                    if not in_args or len(in_args) < 2: raise TypeError("缺少输入参数")
                    a, b = in_args[0], in_args[1]
                    if "add" in op: out_vars[nid] = {0: a + b}
                    elif "sub" in op: out_vars[nid] = {0: a - b}
                    elif "mul" in op: out_vars[nid] = {0: a * b}
                    elif "matmul" in op: out_vars[nid] = {0: torch.matmul(a, b)}
                
                # --- 原生网络层 ---
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
                    
                    if in_args and in_args[0] is not None:
                        sh = in_args[0].shape
                        if "Conv" in ntype and len(sh) >= 2:
                            kwargs["in_channels"] = sh[1]; updated_params[nid] = updated_params.get(nid, {}); updated_params[nid]["in_channels"] = str(sh[1])
                        elif "Linear" in ntype and len(sh) >= 1:
                            kwargs["in_features"] = sh[-1]; updated_params[nid] = updated_params.get(nid, {}); updated_params[nid]["in_features"] = str(sh[-1])
                        elif "BatchNorm" in ntype and len(sh) >= 2:
                            kwargs["num_features"] = sh[1]; updated_params[nid] = updated_params.get(nid, {}); updated_params[nid]["num_features"] = str(sh[1])
                                
                    layer = getattr(nn, ntype)(**kwargs)
                    out_vars[nid] = {0: layer(*[x for x in in_args if x is not None])}

                # 将推演结果写回 UI 界面！
                for c in connections:
                    if c["from"] == nid and c["from_port"] in out_vars.get(nid, {}):
                        shape_results[f"{c['from']}->{c['to']}"] = str(tuple(out_vars[nid][c["from_port"]].shape))

            except TypeError as e:
                for c in connections:
                    if c["from"] == nid: shape_results[f"{c['from']}->{c['to']}"] = "⏳ 等待有效输入..." if "missing" in str(e) else f"❌ {type(e).__name__}"
            except Exception as e:
                for c in connections:
                    if c["from"] == nid: shape_results[f"{c['from']}->{c['to']}"] = f"❌ 计算冲突"

        # 函数执行到底部，返回数据
        ret_nids = [n for n in topo if nodes[n].get("type") == "Return Function"]
        if ret_nids:
            ret_nid = ret_nids[0]
            return [out_vars[ret_nid][i] for i in sorted(out_vars[ret_nid].keys())]
        return []

    # 1. 预执行一遍，让独立模块的内部线亮起来
    for fname in blocks:
        if fname != "main":
            try: eval_block(fname)
            except Exception: pass
            
    # 2. 从主函数启动正式递归执行！这会用真实尺寸重写刚刚的内部线！
    if "main" in blocks:
        try: eval_block("main")
        except Exception: pass

    # 收尾防静默兜底
    for c in connections:
        ck = f"{c['from']}->{c['to']}"
        if ck not in shape_results: shape_results[ck] = "⏳ 连线挂起..."

    return {"shapes": shape_results, "updated_params": updated_params}