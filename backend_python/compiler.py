import re
import ast

def format_param(val_str, p_type):
    val_str = str(val_str).strip()
    if val_str == "None": return "None"
    if p_type == "int":
        try: return str(int(float(val_str)))
        except ValueError: return val_str
    elif p_type == "float": return val_str
    elif p_type == "bool": return "True" if val_str.lower() in ["true", "1", "t", "yes"] else "False"
    elif p_type == "tuple": return f"({val_str})" if not val_str.startswith("(") else val_str
    elif p_type == "string":
        if val_str.isdigit(): return val_str
        try:
            fv = float(val_str)
            return str(int(fv)) if fv.is_integer() else val_str
        except ValueError: pass
        if val_str.startswith(('(', '[', "'", '"')): return val_str
        return f"'{val_str}'"
    return val_str

def clean_name(name): return re.sub(r'\W|^(?=\d)', '_', name)

# ==========================================
# 核心编译引擎 (V4.1 函数绑定解包版)
# ==========================================
def generate_pytorch_code(project_data, main_class_name="MyNetwork"):
    nodes = project_data.get("main", {}).get("nodes", {})
    connections = project_data.get("main", {}).get("connections", [])

    node_var_map = {}
    type_counters = {}
    for nid, info in nodes.items():
        stype = clean_name(info.get("type", "Unknown"))
        base_name = "input" if stype == "Data_Input" else "output" if stype == "Data_Output" else stype
        type_counters[base_name] = type_counters.get(base_name, 0) + 1
        node_var_map[nid] = f"{base_name}_{type_counters[base_name]}"

    # 【核心】：根据 Def Function -> Data Input 连线提取函数参数
    func_blocks = {} # fname: { inputs: [nid], return: nid }
    for nid, info in nodes.items():
        if info["type"] == "Def Function":
            fname = info["params"]["func_name"]["value"]
            if fname not in func_blocks: func_blocks[fname] = {"inputs": [], "return": None}
            # 找到连接到这个 Def 的所有 Data Input
            for conn in connections:
                if conn["from"] == nid and nodes.get(conn["to"], {}).get("type") == "Data Input":
                    func_blocks[fname]["inputs"].append(conn["to"])
        elif info["type"] == "Return Function":
            fname = info["params"]["func_name"]["value"]
            if fname not in func_blocks: func_blocks[fname] = {"inputs": [], "return": None}
            func_blocks[fname]["return"] = nid

    adj_list = {nid: [] for nid in nodes}
    for conn in connections: adj_list[conn["from"]].append(conn)

    def get_subgraph_topo(start_nids):
        reachable = set(); q = list(start_nids)
        while q:
            curr = q.pop(0)
            if curr not in reachable:
                reachable.add(curr)
                for c in adj_list[curr]:
                    if nodes.get(c["from"], {}).get("type") not in ["Weight Init", "Comment", "Def Function"]:
                        q.append(c["to"])
        sub_in_deg = {n: 0 for n in reachable}
        for n in reachable:
            for c in adj_list[n]:
                if c["to"] in reachable and nodes.get(c["from"], {}).get("type") not in ["Weight Init", "Comment", "Def Function"]:
                    sub_in_deg[c["to"]] += 1
        sq = [n for n in reachable if sub_in_deg[n] == 0]
        topo = []
        while sq:
            curr = sq.pop(0); topo.append(curr)
            for c in adj_list[curr]:
                if c["to"] in reachable and nodes.get(c["from"], {}).get("type") not in ["Weight Init", "Comment", "Def Function"]:
                    sub_in_deg[c["to"]] -= 1
                    if sub_in_deg[c["to"]] == 0: sq.append(c["to"])
        return topo

    methods_code = []
    global_init_lines = []
    looped_nodes = set()
    
    init_methods = {}
    for nid, info in nodes.items():
        if info.get("type") == "Weight Init":
            m_val = info.get("params", {}).get("method", {}).get("value", "kaiming_normal_")
            init_methods[nid] = {"method": m_val.split()[0], "mean": info.get("params", {}).get("mean", {}).get("value", "0.0"), "std": info.get("params", {}).get("std", {}).get("value", "1.0")}

    layer_inits = {conn["to"]: init_methods[conn["from"]] for conn in connections if conn["from"] in init_methods}
    
    import torch.nn as nn

    # 先生成所有独立的函数 (保证 main 函数在最后)
    block_names = list(func_blocks.keys())
    if "main" in block_names: 
        block_names.remove("main")
        block_names.append("main") # 把 main 放最后编译

    for fname in block_names:
        block = func_blocks[fname]
        start_nids = block["inputs"]
        if not start_nids: continue # 没有任何输入则不编译此函数
            
        topo = get_subgraph_topo(start_nids)
        
        # 参数按照 Y 坐标或者名字排序，保证函数签名稳定
        start_nids.sort(key=lambda x: nodes[x]["params"].get("arg_name", {}).get("value", "x"))
        arg_names = [nodes[x]["params"].get("arg_name", {}).get("value", "x") for x in start_nids]
        
        def_line = f"    def forward(self, {', '.join(arg_names)}):" if fname == "main" else f"    def {fname}(self, {', '.join(arg_names)}):"
        lines = []
        indent = "        "
        out_vars = {n: {} for n in topo}
        loop_stack = []

        for nid in topo:
            info = nodes[nid]
            ntype = info.get("type", "")
            var_name = f"v_{node_var_map[nid]}"
            
            in_ports = {}
            for conn in connections:
                if conn["to"] == nid and conn["from"] in out_vars and nodes[conn["from"]].get("type") not in ["Weight Init", "Comment", "Def Function"]:
                    in_ports[conn["to_port"]] = out_vars[conn["from"]][conn["from_port"]]
            in_args = [in_ports[i] for i in sorted(in_ports.keys())]

            if ntype == "Data Input":
                arg_name = info.get("params", {}).get("arg_name", {}).get("value", "x")
                lines.append(f"{indent}{var_name} = {arg_name}")
                out_vars[nid][0] = var_name
                
            elif ntype == "Return Function":
                ret_vars = [in_args[i] for i in range(len(in_args)) if in_args[i] is not None]
                lines.append(f"{indent}return {', '.join(ret_vars) if ret_vars else 'None'}")
                
            elif ntype == "Call Function":
                c_b_name = info["params"].get("func_name", {}).get("value", "my_func")
                out_cnt = int(float(info["params"].get("output_count", {}).get("value", 1)))
                
                if out_cnt > 1:
                    ret_names = [f"{var_name}_{i}" for i in range(out_cnt)]
                    lines.append(f"{indent}{', '.join(ret_names)} = self.{c_b_name}({', '.join(in_args)})")
                    for i in range(out_cnt): out_vars[nid][i] = ret_names[i]
                else:
                    lines.append(f"{indent}{var_name} = self.{c_b_name}({', '.join(in_args)})")
                    out_vars[nid][0] = var_name
                
            elif ntype == "Loop Begin":
                iters = int(float(info["params"].get("iterations", {}).get("value", 3)))
                loop_name = node_var_map[nid]
                loop_var = f"v_loop_{loop_name}"
                lines.append(f"{indent}{loop_var} = {in_args[0] if in_args else 'None'}")
                lines.append(f"{indent}for idx_{loop_name} in range({iters}):")
                indent += "    "
                out_vars[nid][0] = loop_var
                loop_stack.append((loop_name, iters))
                
            elif ntype == "Loop End":
                if loop_stack:
                    curr_loop_name, _ = loop_stack.pop()
                    curr_loop_var = f"v_loop_{curr_loop_name}"
                    lines.append(f"{indent}{curr_loop_var} = {in_args[0] if in_args else 'None'}")
                    indent = indent[:-4]
                    out_vars[nid][0] = curr_loop_var
                    
            elif hasattr(nn, ntype):
                args = [f"{k}={format_param(v.get('value', v.get('default', '')), v.get('type', 'string'))}" for k, v in info.get("params", {}).items() if str(v.get("value", "")) not in ["", "None"]]
                fn_name = f"fn_{node_var_map[nid]}"
                if loop_stack:
                    loop_name, iters = loop_stack[-1]
                    global_init_lines.append(f"        self.{fn_name} = nn.ModuleList([nn.{ntype}({', '.join(args)}) for _ in range({iters})])")
                    lines.append(f"{indent}{var_name} = self.{fn_name}[idx_{loop_name}]({', '.join(in_args)})")
                    looped_nodes.add(nid)
                else:
                    global_init_lines.append(f"        self.{fn_name} = nn.{ntype}({', '.join(args)})")
                    lines.append(f"{indent}{var_name} = self.{fn_name}({', '.join(in_args)})")
                out_vars[nid][0] = var_name
                
            elif ntype == "Reshape": lines.append(f"{indent}{var_name} = {in_args[0]}.view({info['params'].get('shape', {}).get('value', '(1, -1)')})"); out_vars[nid][0] = var_name
            elif ntype == "Concat": lines.append(f"{indent}{var_name} = torch.cat([{', '.join(in_args)}], dim={int(float(info['params'].get('dim', {}).get('value', '-1')))})"); out_vars[nid][0] = var_name
            elif ntype == "Binary Math":
                op = info["params"].get("op", {}).get("value", "add")
                a, b = in_args[0] if len(in_args)>0 else "None", in_args[1] if len(in_args)>1 else "None"
                if "add" in op: lines.append(f"{indent}{var_name} = {a} + {b}")
                elif "sub" in op: lines.append(f"{indent}{var_name} = {a} - {b}")
                elif "mul" in op: lines.append(f"{indent}{var_name} = {a} * {b}")
                elif "matmul" in op: lines.append(f"{indent}{var_name} = torch.matmul({a}, {b})")
                out_vars[nid][0] = var_name

        methods_code.append(def_line + "\n" + "\n".join(lines))

    code = f"import torch\nimport torch.nn as nn\n\nclass {main_class_name}(nn.Module):\n    def __init__(self):\n        super({main_class_name}, self).__init__()\n"
    code += "\n".join(global_init_lines) + "\n" if global_init_lines else "        pass\n"
    
    init_weight_lines = []
    for target_nid, cfg in layer_inits.items():
        if target_nid in node_var_map:
            fn_name = f"fn_{node_var_map[target_nid]}"
            if target_nid in looped_nodes:
                init_weight_lines.append(f"        for layer in self.{fn_name}:")
                if cfg["method"] == "normal_": init_weight_lines.append(f"            nn.init.normal_(layer.weight, mean={cfg['mean']}, std={cfg['std']})")
                else: init_weight_lines.append(f"            nn.init.{cfg['method']}(layer.weight)")
                init_weight_lines.append(f"            if getattr(layer, 'bias', None) is not None:\n                nn.init.zeros_(layer.bias)")
            else:
                if cfg["method"] == "normal_": init_weight_lines.append(f"        nn.init.normal_(self.{fn_name}.weight, mean={cfg['mean']}, std={cfg['std']})")
                else: init_weight_lines.append(f"        nn.init.{cfg['method']}(self.{fn_name}.weight)")
                init_weight_lines.append(f"        if getattr(self.{fn_name}, 'bias', None) is not None:\n            nn.init.zeros_(self.{fn_name}.bias)")

    if init_weight_lines: code += "\n        self._initialize_weights()\n"
    code += "\n"
    code += "\n\n".join(methods_code) + "\n"
    
    if init_weight_lines:
        code += "\n    def _initialize_weights(self):\n" + "\n".join(init_weight_lines) + "\n"
    return code


def generate_train_code(project_data, main_class_name="MyNetwork"):
    nodes = project_data.get("main", {}).get("nodes", {})
    loss_node = optim_node = dataset_node = target_node = config_node = None
    for nid, info in nodes.items():
        lt = info.get("type", "")
        if "Loss" in lt: loss_node = info
        elif lt in ["Adadelta", "Adagrad", "Adam", "AdamW", "SGD"]: optim_node = info
        elif lt == "Dataset Loader": dataset_node = info
        elif lt == "Target Loader": target_node = info
        elif lt == "Training Config": config_node = info

    model_code = generate_pytorch_code(project_data, main_class_name)
    ep = config_node["params"]["epochs"]["value"] if config_node else "100"
    bs = config_node["params"]["batch_size"]["value"] if config_node else "32"
    sf = config_node["params"]["save_freq"]["value"] if config_node else "10"
    sp = config_node["params"]["save_path"]["value"] if config_node else "./weights.pth"
    dp = dataset_node["params"]["dataset_path"]["value"] if dataset_node else "./data"
    d_c = dataset_node["params"]["custom_code"]["value"] if dataset_node else "def get_dataloader(path, batch_size):\n    pass"
    t_c = target_node["params"]["custom_code"]["value"] if target_node else "def process_target(targets):\n    return targets"
    l_t = loss_node["type"] if loss_node else "CrossEntropyLoss"
    l_a = [f"{k}={format_param(v.get('value', ''), v.get('type', 'string'))}" for k, v in loss_node.get("params", {}).items() if str(v.get("value", "")) not in ["", "None"]] if loss_node else []
    o_t = optim_node["type"] if optim_node else "Adam"
    o_a = [f"{k}={format_param(v.get('value', ''), v.get('type', 'string'))}" for k, v in optim_node.get("params", {}).items() if str(v.get("value", "")) not in ["", "None"]] if optim_node else []

    return f"""{model_code}\nimport torch.optim as optim\n{d_c}\n\n{t_c}\n
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = {main_class_name}().to(device)
    dataloader = get_dataloader(r'{dp}', {bs})
    optimizer = optim.{o_t}(model.parameters(), {', '.join(o_a)})
    criterion = nn.{l_t}({', '.join(l_a)})
    for epoch in range({ep}):
        model.train()
        total_loss = 0.0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            loss = criterion(model(inputs.to(device)), process_target(targets.to(device)))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{{epoch+1}}/{ep}] Loss: {{total_loss / len(dataloader):.6f}}")
        if (epoch + 1) % {sf} == 0: torch.save(model.state_dict(), r'{sp}')
if __name__ == '__main__': train()"""

def generate_test_code(project_data, main_class_name="MyNetwork"):
    nodes = project_data.get("main", {}).get("nodes", {})
    config_node = next((info for nid, info in nodes.items() if info.get("type", "") == "Inference Config"), None)
    model_code = generate_pytorch_code(project_data, main_class_name)
    wp = config_node["params"]["weights_path"]["value"] if config_node else "./weights/model.pth"
    ds = config_node["params"]["device"]["value"] if config_node else "cuda"
    return f"""{model_code}\n
class {main_class_name}Inference:
    def __init__(self, weights_path=r'{wp}', device='{ds}'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = {main_class_name}()
        try: self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        except: pass
        self.model.to(self.device).eval()
    @torch.no_grad()
    def generate(self, *args):
        inputs = [arg.to(self.device) for arg in args if isinstance(arg, torch.Tensor)]
        return self.model(*inputs)
"""