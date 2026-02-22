# system_nodes.py
SYSTEM_NODES = {
    "System (系统节点)": {
        "Data Input": {
            "name": "Data Input",
            "inputs": [], "main_out": "tensor",
            "params": [
                {"name": "shape", "type": "tuple", "default": "(1, 3, 224, 224)"},
                {"name": "mode", "type": "enum", "options": ["randn", "zeros", "ones"], "default": "randn"}
            ],
            "description": "输入源：定义尺寸和生成模式。"
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

    "Training (训练配置)": {
        "Dataset Loader": {
            "name": "Dataset Loader", "inputs": [], "main_out": "data",
            "params": [
                {"name": "dataset_path", "type": "string", "default": "./dataset/train"},
                {"name": "custom_code", "type": "code", "default": "def get_dataloader(path, batch_size):\n    import torch\n    from torchvision import datasets, transforms\n    from torch.utils.data import DataLoader\n    \n    # 默认：读取图片，自动将子文件夹名作为 target 标签\n    transform = transforms.Compose([\n        transforms.Resize((224, 224)),\n        transforms.ToTensor()\n    ])\n    dataset = datasets.ImageFolder(root=path, transform=transform)\n    return DataLoader(dataset, batch_size=batch_size, shuffle=True)"}
            ],
            "description": "加载图片数据集，输出 data 供模型训练。"
        },
        "Target Loader": {
            "name": "Target Loader", "inputs": [], "main_out": "target",
            "params": [
                {"name": "custom_code", "type": "code", "default": "def process_target(targets):\n    # DataLoader已自动提取了标签，这里可做额外处理(如One-Hot编码)\n    # 如果不需要处理，直接返回即可\n    return targets"}
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
            "description": "循环节点。将内部包裹的层堆叠执行 N 次。"
        }
    },
    "Math & Ops (运算)": {
        "Concat": {
            "name": "Concat",
            "inputs": ["tensor_a", "tensor_b"], 
            "main_out": "out",
            "params": [{"name": "dim", "type": "int", "default": 1}],
            "description": "在指定维度拼接两个张量。"
        },
        "Math": {
            "name": "Math",
            "inputs": ["tensor_a", "tensor_b"], 
            "main_out": "out",
            "params": [
                {"name": "op", "type": "enum", "options": ["add (+)", "sub (-)", "mul (*)", "div (/)", "matmul"], "default": "add (+)"}
            ],
            "description": "算术运算。"
        },
        "Value Display": {
            "name": "Value Display",
            "inputs": ["tensor"], "main_out": "tensor",
            "params": [
                {"name": "index", "type": "tuple", "default": "(0, 0, 0, 0)"},
                {"name": "result", "type": "string", "default": "Waiting..."}
            ],
            "description": "数值探针：查看特定索引下的标量值。"
        }
    }
}