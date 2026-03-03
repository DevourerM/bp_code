import re

# ==========================================
# 智能参数格式化引擎 (V1.2 类型安全强化版)
# ==========================================
def format_param(val_str, p_type):
    """将前端传来的参数安全、智能地格式化为合法的 Python 代码"""
    val_str = str(val_str).strip()
    
    if val_str == "None":
        return "None"
        
    # 1. 强转整数 (修复: dilation=1.0 -> 1)
    if p_type == "int":
        try:
            return str(int(float(val_str)))
        except ValueError:
            return val_str
            
    # 2. 浮点数
    elif p_type == "float":
        return val_str
        
    # 3. 布尔值
    elif p_type == "bool":
        return "True" if val_str.lower() in ["true", "1", "t", "yes"] else "False"
        
    # 4. 元组
    elif p_type == "tuple":
        if not val_str.startswith("("): 
            val_str = f"({val_str})"
        return val_str
        
    # 5. 智能字符串 (核心修复点)
    elif p_type == "string":
        # 如果是纯数字（如 "3"），说明它是被误判为 string 的必填 int 参数（如 in_channels）
        if val_str.isdigit():
            return val_str
            
        # 如果是浮点数格式（如 "3.0"），且本质是整数，转成整数 "3"
        try:
            f_val = float(val_str)
            if f_val.is_integer():
                return str(int(f_val))
            return val_str
        except ValueError:
            pass
            
        # 如果是一个元组或列表格式（如 "(3, 3)" 或 "[1, 2]"）
        if (val_str.startswith('(') and val_str.endswith(')')) or \
           (val_str.startswith('[') and val_str.endswith(']')):
            return val_str
            
        # 如果前端传过来已经自带引号了，直接返回
        if (val_str.startswith("'") and val_str.endswith("'")) or \
           (val_str.startswith('"') and val_str.endswith('"')):
            return val_str
            
        # 真正的纯文本字符串（如 padding_mode='zeros'），为其加上引号
        return f"'{val_str}'"

    return val_str

def clean_name(name):
    """清理名称，移除特殊字符，确保是合法的 Python 变量名"""
    return re.sub(r'\W|^(?=\d)', '_', name)


# ==========================================
# 核心网络代码生成器
# ==========================================
def generate_pytorch_code(project_data, main_class_name="MyNetwork"):
    model_graphs = project_data
    main_graph = model_graphs.get("main", {"nodes": {}, "connections": []})
    nodes = main_graph.get("nodes", {})
    connections = main_graph.get("connections", [])

    node_var_map = {}
    type_counters = {}

    for nid, info in nodes.items():
        ntype = info.get("type", "Unknown")
        safe_type = clean_name(ntype)
        
        if safe_type == "Data_Input":
            base_name = "input"
        elif safe_type == "Data_Output":
            base_name = "output"
        else:
            base_name = safe_type

        if base_name not in type_counters:
            type_counters[base_name] = 1
        else:
            type_counters[base_name] += 1
            
        node_var_map[nid] = f"{base_name}_{type_counters[base_name]}"

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

    init_lines = []
    forward_lines = []
    
    input_nodes = [nid for nid in topo_order if nodes[nid].get("type") == "Data Input"]
    forward_args = ["self"] + [f"x_{i}" for i in range(len(input_nodes))]
    
    out_vars = {nid: {} for nid in nodes}
    input_idx = 0

    for nid in topo_order:
        info = nodes[nid]
        ntype = info.get("type", "")
        params = info.get("params", {})
        readable_name = node_var_map[nid]  
        
        args = []
        for k, v in params.items():
            val = v.get("value", v.get("default", ""))
            if val != "" and str(val) != "None":
                fmt = format_param(val, v.get("type", "string"))
                if fmt: args.append(f"{k}={fmt}")
                
        args_str = ", ".join(args)
        
        if ntype == "Data Input":
            var_name = f"v_{readable_name}" 
            forward_lines.append(f"        {var_name} = x_{input_idx}")
            out_vars[nid][0] = var_name
            input_idx += 1
            
        elif ntype == "Data Output":
            pass 
            
        elif ntype == "Group":
            sub_name = info.get("name", "SubGraph")
            fn_name = f"fn_{readable_name}"
            var_name = f"v_{readable_name}"
            
            init_lines.append(f"        self.{fn_name} = {sub_name}()")
            
            in_ports = []
            for conn in connections:
                if conn["to"] == nid:
                    in_ports.append((conn["to_port"], conn["from"], conn["from_port"]))
            in_ports.sort(key=lambda x: x[0])
            in_args = [out_vars[f_n][f_p] for _, f_n, f_p in in_ports]
            
            forward_lines.append(f"        {var_name} = self.{fn_name}({', '.join(in_args)})")
            out_vars[nid][0] = var_name
            
        else:
            fn_name = f"fn_{readable_name}" 
            var_name = f"v_{readable_name}" 
            
            init_lines.append(f"        self.{fn_name} = nn.{ntype}({args_str})")
            
            in_ports = {}
            for conn in connections:
                if conn["to"] == nid:
                    in_ports[conn["to_port"]] = out_vars[conn["from"]][conn["from_port"]]
            
            in_args = [in_ports[i] for i in sorted(in_ports.keys())]
            
            if not in_args:
                forward_lines.append(f"        {var_name} = self.{fn_name}()")
            else:
                forward_lines.append(f"        {var_name} = self.{fn_name}({', '.join(in_args)})")
                
            out_vars[nid][0] = var_name

    output_nodes = [nid for nid in topo_order if nodes[nid].get("type") == "Data Output"]
    return_vars = []
    for nid in output_nodes:
        for conn in connections:
            if conn["to"] == nid:
                return_vars.append(out_vars[conn["from"]][conn["from_port"]])

    if not return_vars:
        forward_lines.append("        return None")
    else:
        forward_lines.append(f"        return {', '.join(return_vars)}")

    code = f"import torch\nimport torch.nn as nn\n\n"
    code += f"class {main_class_name}(nn.Module):\n"
    code += "    def __init__(self):\n"
    code += f"        super({main_class_name}, self).__init__()\n"
    
    if init_lines:
        code += "\n".join(init_lines) + "\n"
    else:
        code += "        pass\n"
        
    code += "\n"
    code += f"    def forward({', '.join(forward_args)}):\n"
    code += "\n".join(forward_lines) + "\n"

    return code


# ==========================================
# 训练专用 PyTorch 代码编译器
# ==========================================
def generate_train_code(project_data, main_class_name="MyNetwork"):
    train_graph = project_data.get("main", {})
    nodes = train_graph.get("nodes", {})

    model_node_name = None
    loss_node = None
    optim_node = None
    dataset_node = None
    target_node = None
    config_node = None

    for nid, info in nodes.items():
        l_type = info.get("type", "")
        if l_type == "Group": model_node_name = info.get("name", nid) 
        elif "Loss" in l_type: loss_node = info
        elif l_type in ["Adadelta", "Adagrad", "Adam", "AdamW", "SGD", "RMSprop"]: optim_node = info
        elif l_type == "Dataset Loader": dataset_node = info
        elif l_type == "Target Loader": target_node = info
        elif l_type == "Training Config": config_node = info

    if not model_node_name:
        raise ValueError("训练画布上未检测到导入的网络模型！请先导入 .bpnn 模型并进行连线。")

    model_graphs = {}
    if model_node_name in project_data:
        model_graphs["main"] = project_data[model_node_name]
        for k in project_data.keys():
            if k.startswith(model_node_name + "_"):
                model_graphs[k] = project_data[k]
    else:
        model_graphs["main"] = {"nodes": {}, "connections": []}
        
    model_code = generate_pytorch_code(model_graphs, main_class_name)

    epochs = config_node["params"]["epochs"]["value"] if config_node else "100"
    batch_size = config_node["params"]["batch_size"]["value"] if config_node else "32"
    save_freq = config_node["params"]["save_freq"]["value"] if config_node else "10"
    save_path = config_node["params"]["save_path"]["value"] if config_node else "./weights.pth"
    dataset_path = dataset_node["params"]["dataset_path"]["value"] if dataset_node else "./data"
    
    data_code = dataset_node["params"]["custom_code"]["value"] if dataset_node else "def get_dataloader(path, batch_size):\n    pass"
    target_code = target_node["params"]["custom_code"]["value"] if target_node else "def process_target(targets):\n    return targets"

    loss_type = loss_node["type"] if loss_node else "CrossEntropyLoss"
    loss_args = []
    if loss_node:
        for k, v in loss_node.get("params", {}).items():
            val = v.get("value", "")
            if val != "" and str(val) != "None":
                fmt = format_param(val, v.get("type", "string"))
                if fmt: loss_args.append(f"{k}={fmt}")
    
    optim_type = optim_node["type"] if optim_node else "Adam"
    optim_args = []
    if optim_node:
        for k, v in optim_node.get("params", {}).items():
            val = v.get("value", "")
            if val != "" and str(val) != "None":
                fmt = format_param(val, v.get("type", "string"))
                if fmt: optim_args.append(f"{k}={fmt}")

    train_script = f"""{model_code}
import torch.optim as optim

# ==========================================
# 1. 数据集加载与预处理模块
# ==========================================
{data_code}

{target_code}

# ==========================================
# 2. 训练主循环
# ==========================================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"正在使用 {{device}} 准备训练...")
    
    model = {main_class_name}().to(device)
    
    dataloader = get_dataloader(r'{dataset_path}', {batch_size})
    optimizer = optim.{optim_type}(model.parameters(), {', '.join(optim_args)})
    criterion = nn.{loss_type}({', '.join(loss_args)})

    print("🚀 开始训练...")
    for epoch in range({epochs}):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            targets = process_target(targets)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{{epoch+1}}/{epochs}] Loss: {{avg_loss:.6f}}")

        if (epoch + 1) % {save_freq} == 0:
            import torch
            torch.save(model.state_dict(), r'{save_path}')
            print(f"✅ 阶段权重已保存至 {save_path}")

if __name__ == '__main__':
    train()
"""
    return train_script


# ==========================================
# 推理部署专用 PyTorch 代码编译器
# ==========================================
def generate_test_code(project_data, main_class_name="MyNetwork"):
    test_graph = project_data.get("main", {})
    nodes = test_graph.get("nodes", {})

    model_node_name = None
    config_node = None

    for nid, info in nodes.items():
        l_type = info.get("type", "")
        if l_type == "Group": model_node_name = info.get("name", nid) 
        elif l_type == "Inference Config": config_node = info

    if not model_node_name:
        raise ValueError("部署画布上未检测到导入的网络模型！请先导入 .bpnn 模型并进行连线。")

    model_graphs = {}
    if model_node_name in project_data:
        model_graphs["main"] = project_data[model_node_name]
        for k in project_data.keys():
            if k.startswith(model_node_name + "_"):
                model_graphs[k] = project_data[k]
    else:
        model_graphs["main"] = {"nodes": {}, "connections": []}
        
    model_code = generate_pytorch_code(model_graphs, main_class_name)

    weights_path = config_node["params"]["weights_path"]["value"] if config_node else "./weights/model.pth"
    device_str = config_node["params"]["device"]["value"] if config_node else "cuda"
    
    inference_class_name = f"{main_class_name}Inference"

    test_script = f"""{model_code}

# ==========================================
# 推理部署 API 类
# ==========================================
class {inference_class_name}:
    def __init__(self, weights_path=r'{weights_path}', device='{device_str}'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"初始化推理引擎，运行设备: {{self.device}}")

        self.model = {main_class_name}()

        try:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            print("✅ 预训练权重加载成功！")
        except Exception as e:
            print(f"⚠️ 权重加载失败，将使用随机初始化权重。错误信息: {{e}}")

        self.model.to(self.device)
        self.model.eval() 

    @torch.no_grad()
    def generate(self, input_data):
        \"\"\"
        执行神经网络推理
        :param input_data: 输入的 Tensor 数据
        :return: 网络的预测输出
        \"\"\"
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.to(self.device)

        output = self.model(input_data)
        return output

if __name__ == '__main__':
    print("--- 部署模块连通性测试 ---")
    api = {inference_class_name}()
    
    dummy_input = torch.randn(1, 3, 224, 224).to(api.device)
    result = api.generate(dummy_input)
    print("预测输出尺寸:", result.shape)
"""
    return test_script