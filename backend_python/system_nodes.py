# system_nodes.py
SYSTEM_NODES = {
    "System (系统节点)": {
        "Data Input": {
            "name": "Data Input",
            "inputs": [], "main_out": "tensor",
            "params": [
                {"name": "shape", "type": "tuple", "default": "(1, 3, 224, 224)"},
                {"name": "mode", "type": "enum", "options": ["randn", "zeros", "ones", "randint"], "default": "randn"}
            ],
            "description": "输入源：定义尺寸和生成模式。NLP任务中常用randint生成Token IDs。"
        },
        "Data Output": {
            "name": "Data Output",
            "inputs": ["tensor"], "main_out": "",
            "params": [],
            "description": "输出端：查看最终结果形状。"
        },
        "Comment (注释)": {
            "name": "Comment", "inputs": [], "main_out": "",
            "params": [
                {"name": "text", "type": "string", "default": "【在此放置模型】"}
            ],
            "description": "在画布上添加备注信息，不影响代码编译。"
        }
    },

    "Structure (架构)": {
        "Group": {
            "name": "Group", 
            "inputs": [], 
            "main_out": "out",
            "params": [
                {"name": "input_count", "type": "int", "default": 1}
            ],
            "description": "封装节点。点击进入子空间，调整 input_count 增加多输入。"
        },
        "Loop": {
            "name": "Loop", 
            "inputs": ["in"], 
            "main_out": "out",
            "params": [
                {"name": "iterations", "type": "int", "default": 3}
            ],
            "description": "循环节点。将内部包裹的层堆叠执行 N 次 (如Transformer的N层Block)。"
        }
    },

    "Tensor Shape (张量形态)": {
        "Reshape (重塑)": {
            "name": "Reshape",
            "inputs": ["tensor"], "main_out": "out",
            "params": [
                {"name": "shape", "type": "tuple", "default": "(1, -1)"}
            ],
            "description": "改变张量形状。-1代表自动推断。注意力机制拆分多头必备。"
        },
        "Permute (转置/置换)": {
            "name": "Permute",
            "inputs": ["tensor"], "main_out": "out",
            "params": [
                {"name": "dims", "type": "tuple", "default": "(0, 2, 1, 3)"}
            ],
            "description": "重排张量的维度。例如将 (B, Seq, H, D) 转为 (B, H, Seq, D)。"
        },
        "Slice (切片)": {
            "name": "Slice",
            "inputs": ["tensor"], "main_out": "out",
            "params": [
                {"name": "dim", "type": "int", "default": 1},
                {"name": "start", "type": "int", "default": 0},
                {"name": "end", "type": "string", "default": "None"},
                {"name": "step", "type": "int", "default": 1}
            ],
            "description": "提取张量的某一部分，类似于 tensor[:, start:end:step, ...]。"
        },
        "Unsqueeze (升维)": {
            "name": "Unsqueeze",
            "inputs": ["tensor"], "main_out": "out",
            "params": [{"name": "dim", "type": "int", "default": 1}],
            "description": "在指定位置增加一个大小为1的维度。"
        },
        "Squeeze (降维)": {
            "name": "Squeeze",
            "inputs": ["tensor"], "main_out": "out",
            "params": [{"name": "dim", "type": "string", "default": "None"}],
            "description": "移除大小为1的维度。如果不填，则移除所有为1的维度。"
        },
        "Expand (扩展/广播)": {
            "name": "Expand",
            "inputs": ["tensor"], "main_out": "out",
            "params": [{"name": "sizes", "type": "tuple", "default": "(1, 8, 8)"}],
            "description": "将大小为1的维度广播扩展到指定大小，不额外消耗内存。"
        }
    },

    "Math & Ops (数学运算)": {
        "Binary Math (双目运算)": {
            "name": "Binary Math",
            "inputs": ["tensor_a", "tensor_b"], "main_out": "out",
            "params": [
                {"name": "op", "type": "enum", "options": ["add (+)", "sub (-)", "mul (*)", "div (/)", "matmul (@)", "pow (**)"], "default": "matmul (@)"}
            ],
            "description": "双输入数学运算。matmul为矩阵乘法，是构建Transformer的核心。"
        },
        "Unary Math (单目运算)": {
            "name": "Unary Math",
            "inputs": ["tensor"], "main_out": "out",
            "params": [
                {"name": "op", "type": "enum", "options": ["exp", "log", "sqrt", "abs", "sin", "cos"], "default": "sqrt"}
            ],
            "description": "单输入数学运算。如求平方根(用于Attention缩放)、计算正余弦(用于位置编码)。"
        },
        "Reduce (聚合运算)": {
            "name": "Reduce",
            "inputs": ["tensor"], "main_out": "out",
            "params": [
                {"name": "op", "type": "enum", "options": ["mean", "sum", "max", "min"], "default": "mean"},
                {"name": "dim", "type": "int", "default": -1},
                {"name": "keepdim", "type": "bool", "default": "True"}
            ],
            "description": "沿着指定维度进行降维计算(如求均值、求和)。"
        },
        "Concat (拼接)": {
            "name": "Concat",
            "inputs": ["tensor_a", "tensor_b"], "main_out": "out",
            "params": [{"name": "dim", "type": "int", "default": -1}],
            "description": "在指定维度上将两个张量拼接在一起。"
        },
        "Masked Fill (掩码填充)": {
            "name": "Masked Fill",
            "inputs": ["tensor", "mask"], "main_out": "out",
            "params": [
                {"name": "value", "type": "float", "default": -1e9}
            ],
            "description": "将mask为True的位置填充为特定值(-1e9)。实现Causal Attention必备。"
        }
    },

    "Training (训练配置)": {
        "Dataset Loader": {
            "name": "Dataset Loader", "inputs": [], "main_out": "data",
            "params": [
                {"name": "dataset_path", "type": "string", "default": "./dataset/train"},
                {"name": "custom_code", "type": "code", "default": "def get_dataloader(path, batch_size):\n    # 默认返回图像数据，NLP任务请自行修改为TextDataset\n    pass"}
            ],
            "description": "加载数据集，输出 data 供模型训练。"
        },
        "Target Loader": {
            "name": "Target Loader", "inputs": [], "main_out": "target",
            "params": [
                {"name": "custom_code", "type": "code", "default": "def process_target(targets):\n    return targets"}
            ],
            "description": "处理目标标签数据，连接至 Loss 函数。"
        },
        "Training Config": {
            "name": "Training Config", "inputs": ["data"], "main_out": "config",
            "params": [
                {"name": "epochs", "type": "int", "default": 100},
                {"name": "batch_size", "type": "int", "default": 32},
                {"name": "save_freq", "type": "int", "default": 10},
                {"name": "save_path", "type": "string", "default": "./weights/model.pth"}
            ],
            "description": "全局训练超参数，串联在数据流中。"
        }
    },

    "Testing (部署推理)": {
        "Inference Config": {
            "name": "Inference Config", "inputs": [], "main_out": "",
            "params": [
                {"name": "weights_path", "type": "string", "default": "./weights/model.pth"},
                {"name": "device", "type": "enum", "options": ["cuda", "cpu"], "default": "cuda"}
            ],
            "description": "推理配置节点，设置预训练权重路径与运行设备。"
        }
    }
}