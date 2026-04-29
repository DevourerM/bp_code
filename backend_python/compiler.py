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

# WCC 孤岛拆分器
def get_blocks(nodes, connections):
    adj = {n: [] for n in nodes}
    for c in connections:
        if c["from"] in nodes and c["to"] in nodes:
            adj[c["from"]].append(c["to"])
            adj[c["to"]].append(c["from"])
            
    visited = set()
    blocks = {}
    for nid, info in nodes.items():
        if info.get("type") == "Def Function":
            b_name = info.get("params", {}).get("func_name", {}).get("value", "main")
            comp = set()
            q = [nid]
            visited.add(nid)
            while q:
                curr = q.pop(0)
                comp.add(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append(neighbor)
            blocks[b_name] = comp
    return blocks

# ==========================================
# 核心编译引擎 (V6.0 完美多态分离类生成)
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

    blocks = get_blocks(nodes, connections)
    
    init_methods = {}
    for nid, info in nodes.items():
        if info.get("type") == "Weight Init":
            m_val = info.get("params", {}).get("method", {}).get("value", "kaiming_normal_")
            init_methods[nid] = {"method": m_val.split()[0], "mean": info.get("params", {}).get("mean", {}).get("value", "0.0"), "std": info.get("params", {}).get("std", {}).get("value", "1.0")}

    layer_inits = {conn["to"]: init_methods[conn["from"]] for conn in connections if conn["from"] in init_methods}
    
    # 强制将主模型类放最后
    block_names = list(blocks.keys())
    if "main" in block_names: 
        block_names.remove("main")
        block_names.append("main")

    import torch.nn as nn
    generated_classes_code = []

    # 为每一个物理区块生成独立的 nn.Module
    for fname in block_names:
        comp_nodes = blocks[fname]
        cname = main_class_name if fname == "main" else fname
        
        start_nids = [n for n in comp_nodes if nodes[n]["type"] == "Data Input"]
        if not start_nids: continue 
            
        start_nids.sort(key=lambda x: nodes[x]["params"].get("arg_name", {}).get("value", "x"))
        arg_names = [nodes[x]["params"].get("arg_name", {}).get("value", "x") for x in start_nids]
        
        in_deg = {n: 0 for n in comp_nodes}
        dir_adj = {n: [] for n in comp_nodes}
        for c in connections:
            fn, tn = c["from"], c["to"]
            if fn in comp_nodes and tn in comp_nodes and nodes.get(fn, {}).get("type") not in ["Weight Init", "Comment", "Def Function"]:
                dir_adj[fn].append(tn)
                in_deg[tn] += 1
        q = [n for n in comp_nodes if in_deg[n] == 0]
        topo = []
        while q:
            curr = q.pop(0); topo.append(curr)
            for nbr in dir_adj[curr]:
                in_deg[nbr] -= 1
                if in_deg[nbr] == 0: q.append(nbr)
        
        init_lines = []
        forward_lines = []
        weight_init_lines = []
        indent = "        "
        out_vars = {n: {} for n in topo}
        loop_stack = []
        looped_nodes = set()

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
                forward_lines.append(f"{indent}{var_name} = {arg_name}")
                out_vars[nid][0] = var_name
                
            elif ntype == "Return Function":
                ret_vars = [in_args[i] for i in range(len(in_args)) if in_args[i] is not None]
                forward_lines.append(f"{indent}return {', '.join(ret_vars) if ret_vars else 'None'}")
                
            elif ntype == "Call Function":
                c_b_name = info["params"].get("func_name", {}).get("value", "my_func")
                out_cnt = int(float(info["params"].get("output_count", {}).get("value", 1)))
                fn_name = f"fn_Call_{node_var_map[nid]}"
                
                if loop_stack:
                    loop_name, iters = loop_stack[-1]
                    init_lines.append(f"        self.{fn_name} = nn.ModuleList([{c_b_name}() for _ in range({iters})])")
                    call_str = f"self.{fn_name}[idx_{loop_name}]({', '.join(in_args)})"
                else:
                    init_lines.append(f"        self.{fn_name} = {c_b_name}()")
                    call_str = f"self.{fn_name}({', '.join(in_args)})"
                
                if out_cnt > 1:
                    ret_names = [f"{var_name}_{i}" for i in range(out_cnt)]
                    forward_lines.append(f"{indent}{', '.join(ret_names)} = {call_str}")
                    for i in range(out_cnt): out_vars[nid][i] = ret_names[i]
                else:
                    forward_lines.append(f"{indent}{var_name} = {call_str}")
                    out_vars[nid][0] = var_name
                
            elif ntype == "Loop Begin":
                iters = int(float(info["params"].get("iterations", {}).get("value", 3)))
                loop_name = node_var_map[nid]
                loop_var = f"v_loop_{loop_name}"
                forward_lines.append(f"{indent}{loop_var} = {in_args[0] if in_args else 'None'}")
                forward_lines.append(f"{indent}for idx_{loop_name} in range({iters}):")
                indent += "    "
                out_vars[nid][0] = loop_var
                loop_stack.append((loop_name, iters))
                
            elif ntype == "Loop End":
                if loop_stack:
                    curr_loop_name, _ = loop_stack.pop()
                    curr_loop_var = f"v_loop_{curr_loop_name}"
                    forward_lines.append(f"{indent}{curr_loop_var} = {in_args[0] if in_args else 'None'}")
                    indent = indent[:-4]
                    out_vars[nid][0] = curr_loop_var
                    
            elif hasattr(nn, ntype):
                args = [f"{k}={format_param(v.get('value', v.get('default', '')), v.get('type', 'string'))}" for k, v in info.get("params", {}).items() if str(v.get("value", "")) not in ["", "None"]]
                fn_name = f"fn_{node_var_map[nid]}"
                if loop_stack:
                    loop_name, iters = loop_stack[-1]
                    init_lines.append(f"        self.{fn_name} = nn.ModuleList([nn.{ntype}({', '.join(args)}) for _ in range({iters})])")
                    forward_lines.append(f"{indent}{var_name} = self.{fn_name}[idx_{loop_name}]({', '.join(in_args)})")
                    looped_nodes.add(nid)
                else:
                    init_lines.append(f"        self.{fn_name} = nn.{ntype}({', '.join(args)})")
                    forward_lines.append(f"{indent}{var_name} = self.{fn_name}({', '.join(in_args)})")
                out_vars[nid][0] = var_name
                
            elif ntype == "Reshape": forward_lines.append(f"{indent}{var_name} = {in_args[0]}.view({info['params'].get('shape', {}).get('value', '(1, -1)')})"); out_vars[nid][0] = var_name
            elif ntype == "Concat": forward_lines.append(f"{indent}{var_name} = torch.cat([{', '.join(in_args)}], dim={int(float(info['params'].get('dim', {}).get('value', '-1')))})"); out_vars[nid][0] = var_name
            elif ntype == "Binary Math":
                op = info["params"].get("op", {}).get("value", "add")
                a, b = in_args[0] if len(in_args)>0 else "None", in_args[1] if len(in_args)>1 else "None"
                if "add" in op: forward_lines.append(f"{indent}{var_name} = {a} + {b}")
                elif "sub" in op: forward_lines.append(f"{indent}{var_name} = {a} - {b}")
                elif "mul" in op: forward_lines.append(f"{indent}{var_name} = {a} * {b}")
                elif "matmul" in op: forward_lines.append(f"{indent}{var_name} = torch.matmul({a}, {b})")
                out_vars[nid][0] = var_name

        for target_nid, cfg in layer_inits.items():
            if target_nid in comp_nodes and target_nid in node_var_map:
                fn_name = f"fn_{node_var_map[target_nid]}"
                if target_nid in looped_nodes:
                    weight_init_lines.append(f"        for layer in self.{fn_name}:")
                    if cfg["method"] == "normal_": weight_init_lines.append(f"            nn.init.normal_(layer.weight, mean={cfg['mean']}, std={cfg['std']})")
                    else: weight_init_lines.append(f"            nn.init.{cfg['method']}(layer.weight)")
                    weight_init_lines.append(f"            if getattr(layer, 'bias', None) is not None:\n                nn.init.zeros_(layer.bias)")
                else:
                    if cfg["method"] == "normal_": weight_init_lines.append(f"        nn.init.normal_(self.{fn_name}.weight, mean={cfg['mean']}, std={cfg['std']})")
                    else: weight_init_lines.append(f"        nn.init.{cfg['method']}(self.{fn_name}.weight)")
                    weight_init_lines.append(f"        if getattr(self.{fn_name}, 'bias', None) is not None:\n            nn.init.zeros_(self.{fn_name}.bias)")

        class_str = f"class {cname}(nn.Module):\n"
        class_str += "    def __init__(self):\n"
        class_str += f"        super({cname}, self).__init__()\n"
        class_str += "\n".join(init_lines) + "\n" if init_lines else "        pass\n"
        if weight_init_lines: class_str += "\n        self._initialize_weights()\n"
        
        class_str += f"\n    def forward(self, {', '.join(arg_names)}):\n"
        class_str += "\n".join(forward_lines) + "\n"
        
        if weight_init_lines:
            class_str += "\n    def _initialize_weights(self):\n" + "\n".join(weight_init_lines) + "\n"
            
        generated_classes_code.append(class_str)

    return "import torch\nimport torch.nn as nn\n\n" + "\n\n".join(generated_classes_code)

def generate_train_code(project_data, main_class_name="MyNetwork"):
    nodes = project_data.get("main", {}).get("nodes", {})
    loss_node = optim_node = dataset_node = target_node = config_node = None
    for nid, info in nodes.items():
        lt = info.get("type", "")
        if "Loss" in lt: loss_node = info
        elif lt in ["Adadelta", "Adagrad", "Adam", "AdamW", "SGD", "RMSprop"]: optim_node = info
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
if __name__ == '__main__':
    api = {main_class_name}Inference()
    print("Test instantiated successfully.")
"""