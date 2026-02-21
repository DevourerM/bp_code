# compiler.py
import re
import ast

def clean_name(name):
    # 将节点名称转化为合法的 Python 变量名
    return re.sub(r'\W|^(?=\d)', '_', name)

# ==========================================
# 【核心升级】：智能参数类型格式化器
# ==========================================
def format_param(val_str, p_type):
    val_str = str(val_str).strip()
    
    # 1. 忽略空值和 None
    if val_str == "" or val_str == "None":
        return None
    
    # 2. 处理明确的 Bool 类型
    if p_type == "bool":
        return "True" if val_str.lower() == "true" else "False"
        
    # 3. 处理明确的整数 (解决 SpinBox 传过来 1.0 的问题)
    elif p_type == "int":
        try:
            return str(int(float(val_str)))
        except:
            pass
            
    # 4. 处理明确的浮点数
    elif p_type == "float":
        try:
            return str(float(val_str))
        except:
            pass

    # 5. 【智能嗅探】：尝试将其作为 Python 代码求值
    try:
        # 如果是 "5"，求值后是 int 类型的 5
        # 如果是 "(3, 3)"，求值后是 tuple 类型的 (3, 3)
        evaluated = ast.literal_eval(val_str)
        
        # 如果求值后发现确实是个纯字符串，加上单引号
        if isinstance(evaluated, str):
            return f"'{evaluated}'"
            
        # 否则直接返回它的代码形态
        return str(evaluated)
    except Exception:
        # 6. 如果求值报错 (比如 "zeros", "reflect")，说明它是普通字符串，安全加引号
        return f"'{val_str}'"

def generate_pytorch_code(project_data, main_class_name="MyNetwork"):
    code_blocks = []
    code_blocks.append("import torch")
    code_blocks.append("import torch.nn as nn\n")
    
    # 优先编译子空间，确保它们在主网络之前被定义
    graphs_to_compile = [g for g in project_data.keys() if g != "main"] + ["main"]
    
    for graph_id in graphs_to_compile:
        graph = project_data[graph_id]
        nodes = graph.get("nodes", {})
        conns = graph.get("connections", [])
        
        c_name = main_class_name if graph_id == "main" else f"SubNet_{clean_name(graph_id)}"
        
        init_lines = []
        forward_lines = []
        
        # 1. 解析 Input
        data_inputs = [nid for nid, info in nodes.items() if info["type"] == "Data Input"]
        def sort_key(nid):
            nid_str = str(nid)
            if nid_str.startswith("input") and nid_str[5:].isdigit():
                return int(nid_str[5:])
            return nodes[nid].get("pos_y", 0)
        data_inputs.sort(key=sort_key)
        
        input_args = ["self"]
        for i, nid in enumerate(data_inputs):
            arg_name = f"x_{i}"
            input_args.append(arg_name)
            forward_lines.append(f"        v_{clean_name(nid)} = {arg_name}")
        
        # 2. 拓扑排序
        in_degree = {nid: 0 for nid in nodes}
        adj = {nid: [] for nid in nodes}
        incoming_ports = {nid: {} for nid in nodes}
        
        for c in conns:
            f, t, tp = c["from"], c["to"], c.get("to_port", 0)
            adj[f].append(t)
            in_degree[t] += 1
            incoming_ports[t][tp] = f
            
        q = [nid for nid in nodes if in_degree[nid] == 0]
        topo_order = []
        while q:
            curr = q.pop(0)
            topo_order.append(curr)
            for neighbor in adj[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    q.append(neighbor)
                    
        for nid in nodes:
            if nid not in topo_order: topo_order.append(nid)
                
        # 3. 生成运算代码
        for nid in topo_order:
            info = nodes[nid]
            l_type = info["type"]
            if l_type == "Data Input": continue
            
            # 搜集输入变量
            req_inputs = []
            ins = info.get("inputs", [])
            req_count = len(ins) if ins else (1 if info.get("main_in") else 0)
            for port in range(req_count):
                src_node = incoming_ports[nid].get(port)
                req_inputs.append(f"v_{clean_name(src_node)}" if src_node else "None")
            
            # 【应用参数清洗逻辑】
            p_raw = info.get("params", {})
            params = {}
            for k, v in p_raw.items():
                val_str = v.get("value", "") if isinstance(v, dict) else str(v)
                p_type = v.get("type", "string") if isinstance(v, dict) else "string"
                
                fmt_val = format_param(val_str, p_type)
                if fmt_val is not None:
                    params[k] = fmt_val
            
            out_var = f"v_{clean_name(nid)}"
            
            if l_type == "Data Output":
                ret_val = req_inputs[0] if req_inputs else "None"
                forward_lines.append(f"        return {ret_val}")
                continue
                
            if l_type == "Group":
                sub_class = f"SubNet_{clean_name(nid)}"
                layer_name = f"group_{clean_name(nid)}"
                init_lines.append(f"        self.{layer_name} = {sub_class}()")
                forward_lines.append(f"        {out_var} = self.{layer_name}({', '.join(req_inputs)})")
                
            elif l_type == "Loop":
                sub_class = f"SubNet_{clean_name(nid)}"
                layer_name = f"loop_{clean_name(nid)}"
                iters = params.get("iterations", "3")
                init_lines.append(f"        self.{layer_name} = {sub_class}()")
                forward_lines.append(f"        {out_var} = {req_inputs[0]}")
                forward_lines.append(f"        for _ in range({iters}):")
                forward_lines.append(f"            {out_var} = self.{layer_name}({out_var})")
                
            elif l_type == "Concat":
                dim = params.get("dim", "1")
                forward_lines.append(f"        {out_var} = torch.cat(({', '.join(req_inputs)}), dim={dim})")
                
            elif l_type == "Math":
                op = params.get("op", "'add'").replace("'", "").replace('"', "")
                a, b = req_inputs[0], req_inputs[1] if len(req_inputs) > 1 else "None"
                if "add" in op: forward_lines.append(f"        {out_var} = {a} + {b}")
                elif "sub" in op: forward_lines.append(f"        {out_var} = {a} - {b}")
                elif "mul" in op: forward_lines.append(f"        {out_var} = {a} * {b}")
                elif "div" in op: forward_lines.append(f"        {out_var} = {a} / {b}")
                elif "matmul" in op: forward_lines.append(f"        {out_var} = torch.matmul({a}, {b})")
                
            elif l_type == "Value Display":
                forward_lines.append(f"        {out_var} = {req_inputs[0]}")
                
            else: # 常规 PyTorch 层
                layer_name = f"op_{clean_name(nid)}"
                clean_args = [f"{k}={v}" for k, v in params.items()]
                init_lines.append(f"        self.{layer_name} = nn.{l_type}({', '.join(clean_args)})")
                forward_lines.append(f"        {out_var} = self.{layer_name}({req_inputs[0]})")
                
        # 4. 组装 Python 类
        code_blocks.append(f"class {c_name}(nn.Module):")
        code_blocks.append(f"    def __init__(self):")
        code_blocks.append(f"        super({c_name}, self).__init__()")
        if not init_lines: code_blocks.append(f"        pass")
        else: code_blocks.extend(init_lines)
        code_blocks.append("")
        
        code_blocks.append(f"    def forward({', '.join(input_args)}):")
        if not forward_lines: code_blocks.append(f"        pass")
        else: code_blocks.extend(forward_lines)
        code_blocks.append("\n")
        
    return "\n".join(code_blocks)