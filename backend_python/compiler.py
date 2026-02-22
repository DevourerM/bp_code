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


# 将这段代码追加到 compiler.py 文件的最下方

# ==========================================
# 训练专用 PyTorch 代码编译器
# ==========================================
# ==========================================
# 训练专用 PyTorch 代码编译器
# ==========================================
def generate_train_code(project_data, main_class_name="MyNetwork"):
    train_graph = project_data.get("main", {})
    nodes = train_graph.get("nodes", {})

    # 1. 扫描训练画布，寻找关键组件
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

    # 2. 提取并生成网络结构模型代码
    model_graphs = {}
    if model_node_name in project_data:
        model_graphs["main"] = project_data[model_node_name]
        for k in project_data.keys():
            if k.startswith(model_node_name + "_"):
                model_graphs[k] = project_data[k]
    else:
        model_graphs["main"] = {"nodes": {}, "connections": []}
        
    model_code = generate_pytorch_code(model_graphs, main_class_name)

    # 3. 提取训练超参数和代码块
    epochs = config_node["params"]["epochs"]["value"] if config_node else "100"
    batch_size = config_node["params"]["batch_size"]["value"] if config_node else "32"
    save_freq = config_node["params"]["save_freq"]["value"] if config_node else "10"
    save_path = config_node["params"]["save_path"]["value"] if config_node else "./weights.pth"
    
    dataset_path = dataset_node["params"]["dataset_path"]["value"] if dataset_node else "./data"
    
    # 获取 Data 和 Target 的自定义代码块
    data_code = dataset_node["params"]["custom_code"]["value"] if dataset_node else "def get_dataloader(path, batch_size):\n    pass"
    target_code = target_node["params"]["custom_code"]["value"] if target_node else "def process_target(targets):\n    return targets"

    # 处理 Loss 与 Optimizer 参数
    loss_type = loss_node["type"] if loss_node else "CrossEntropyLoss"
    loss_args = []
    if loss_node:
        for k, v in loss_node.get("params", {}).items():
            val = v.get("value", "")
            if val != "" and val != "None":
                from compiler import format_param
                fmt = format_param(val, v.get("type", "string"))
                if fmt: loss_args.append(f"{k}={fmt}")
    
    optim_type = optim_node["type"] if optim_node else "Adam"
    optim_args = []
    if optim_node:
        for k, v in optim_node.get("params", {}).items():
            val = v.get("value", "")
            if val != "" and val != "None":
                from compiler import format_param
                fmt = format_param(val, v.get("type", "string"))
                if fmt: optim_args.append(f"{k}={fmt}")

    # 4. 组装终极可运行的 train.py 代码
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
    
    # 初始化模型
    model = {main_class_name}().to(device)
    
    # 准备数据、优化器和损失函数
    dataloader = get_dataloader(r'{dataset_path}', {batch_size})
    optimizer = optim.{optim_type}(model.parameters(), {', '.join(optim_args)})
    criterion = nn.{loss_type}({', '.join(loss_args)})

    print("🚀 开始训练...")
    for epoch in range({epochs}):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # 将数据和标签转移到设备
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # 对标签进行进一步处理 (根据 Target Loader 的逻辑)
            targets = process_target(targets)

            # 前向传播与误差计算
            optimizer.zero_grad()
            outputs = model(inputs) # Dataset 流入模型
            loss = criterion(outputs, targets) # 模型预测值与 Target 的误差对比
            
            # 反向传播与优化
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 打印日志
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{{epoch+1}}/{epochs}] Loss: {{avg_loss:.6f}}")

        # 定期保存权重
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

    # 提取网络结构
    model_graphs = {}
    if model_node_name in project_data:
        model_graphs["main"] = project_data[model_node_name]
        for k in project_data.keys():
            if k.startswith(model_node_name + "_"):
                model_graphs[k] = project_data[k]
    else:
        model_graphs["main"] = {"nodes": {}, "connections": []}
        
    model_code = generate_pytorch_code(model_graphs, main_class_name)

    # 提取推理配置
    weights_path = config_node["params"]["weights_path"]["value"] if config_node else "./weights/model.pth"
    device_str = config_node["params"]["device"]["value"] if config_node else "cuda"
    
    # 包装类名 (如 ResNet -> ResNetInference)
    inference_class_name = f"{main_class_name}Inference"

    # 组装极简推理 API
    test_script = f"""{model_code}

# ==========================================
# 推理部署 API 类
# ==========================================
class {inference_class_name}:
    def __init__(self, weights_path=r'{weights_path}', device='{device_str}'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"初始化推理引擎，运行设备: {{self.device}}")

        # 初始化网络结构
        self.model = {main_class_name}()

        # 加载预训练权重
        try:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            print("✅ 预训练权重加载成功！")
        except Exception as e:
            print(f"⚠️ 权重加载失败，将使用随机初始化权重。错误信息: {{e}}")

        self.model.to(self.device)
        self.model.eval() # 开启推理模式，冻结 Dropout 和 BatchNorm

    @torch.no_grad()
    def generate(self, input_data):
        \"\"\"
        执行神经网络推理
        :param input_data: 输入的 Tensor 数据
        :return: 网络的预测输出
        \"\"\"
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.to(self.device)

        # 前向传播推理
        output = self.model(input_data)
        return output

if __name__ == '__main__':
    # 快速测试代码
    print("--- 部署模块连通性测试 ---")
    api = {inference_class_name}()
    
    # 构建伪输入 (尺寸由你在蓝图中定义)
    dummy_input = torch.randn(1, 3, 224, 224).to(api.device)
    result = api.generate(dummy_input)
    print("预测输出尺寸:", result.shape)
"""
    return test_script