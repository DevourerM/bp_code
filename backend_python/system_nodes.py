# system_nodes.py

SYSTEM_NODES = {
    "System (系统节点)": {
        "Def Function": {
            "name": "Def Function", "inputs": [], "main_out": "def",
            "params": [{"name": "func_name", "type": "string", "default": "main"}],
            "description": "函数定义起点。将其连接到多个 Data Input 以绑定函数入参。"
        },
        "Return Function": {
            "name": "Return Function", "inputs": [], "main_out": "",
            "params": [
                {"name": "input_count", "type": "int", "default": 1}
            ],
            "description": "函数终点。将最终的计算结果连接至此。不需要命名，系统会自动溯源。"
        },
        "Data Input": {
            "name": "Data Input", "inputs": ["def"], "main_out": "tensor",
            "params": [
                {"name": "arg_name", "type": "string", "default": "x"},
                {"name": "shape", "type": "tuple", "default": "(1, 3, 224, 224)"}
            ],
            "description": "函数的输入参数。必须连接在 Def Function 后面。自带期望形状仅作独立开发时的预览参照。"
        },
        "Call Function": {
            "name": "Call Function", "inputs": [], "main_out": "",
            "params": [
                {"name": "func_name", "type": "string", "default": "my_func"},
                {"name": "input_count", "type": "int", "default": 1},
                {"name": "output_count", "type": "int", "default": 1}
            ],
            "description": "调用在画布上定义的其他模块。它会自动将输入尺寸代入目标模块进行递归推导。"
        },
        "Loop Begin": {
            "name": "Loop Begin", "inputs": ["tensor"], "main_out": "tensor",
            "params": [{"name": "iterations", "type": "int", "default": 3}],
            "description": "循环起点。中间包裹的网络层将被动态转换为 nn.ModuleList 独立执行。"
        },
        "Loop End": {
            "name": "Loop End", "inputs": ["tensor"], "main_out": "tensor",
            "params": [], "description": "循环终点。"
        },
        "Comment (注释)": {
            "name": "Comment", "inputs": [], "main_out": "",
            "params": [{"name": "text", "type": "string", "default": "【说明】"}],
            "description": "在画布上添加备注信息。"
        }
    },

    "Tensor Shape (张量形态)": {
        "Reshape (重塑)": {"name": "Reshape", "inputs": ["tensor"], "main_out": "out", "params": [{"name": "shape", "type": "tuple", "default": "(1, -1)"}], "description": "改变张量形状。-1代表自动推断。"},
        "Permute (转置)": {"name": "Permute", "inputs": ["tensor"], "main_out": "out", "params": [{"name": "dims", "type": "tuple", "default": "(0, 2, 1, 3)"}], "description": "重排张量维度。"},
        "Squeeze (降维)": {"name": "Squeeze", "inputs": ["tensor"], "main_out": "out", "params": [{"name": "dim", "type": "string", "default": "None"}], "description": "移除大小为1的维度。"},
        "Unsqueeze (升维)": {"name": "Unsqueeze", "inputs": ["tensor"], "main_out": "out", "params": [{"name": "dim", "type": "int", "default": 1}], "description": "增加一个大小为1的维度。"},
        "Expand (扩展)": {"name": "Expand", "inputs": ["tensor"], "main_out": "out", "params": [{"name": "sizes", "type": "tuple", "default": "(1, 8, 8)"}], "description": "广播扩展维度。"}
    },

    "Math & Ops (数学运算)": {
        "Binary Math": {"name": "Binary Math", "inputs": ["tensor_a", "tensor_b"], "main_out": "out", "params": [{"name": "op", "type": "enum", "options": ["add (+)", "sub (-)", "mul (*)", "div (/)", "matmul (@)"], "default": "add (+)"}], "description": "双输入数学运算。"},
        "Concat": {"name": "Concat", "inputs": ["tensor_a", "tensor_b"], "main_out": "out", "params": [{"name": "dim", "type": "int", "default": -1}], "description": "拼接张量。"}
    },

    "Training (训练配置)": {
        "Dataset Loader": {"name": "Dataset Loader", "inputs": [], "main_out": "data", "params": [{"name": "dataset_path", "type": "string", "default": "./dataset/train"}, {"name": "custom_code", "type": "code", "default": "def get_dataloader(path, batch_size):\n    pass"}], "description": "加载数据集。"},
        "Target Loader": {"name": "Target Loader", "inputs": [], "main_out": "target", "params": [{"name": "custom_code", "type": "code", "default": "def process_target(targets):\n    return targets"}], "description": "处理目标标签。"},
        "Training Config": {"name": "Training Config", "inputs": ["data"], "main_out": "config", "params": [{"name": "epochs", "type": "int", "default": 100}, {"name": "batch_size", "type": "int", "default": 32}, {"name": "save_freq", "type": "int", "default": 10}, {"name": "save_path", "type": "string", "default": "./weights/model.pth"}], "description": "全局训练配置。"},
        "Weight Init": {"name": "Weight Init", "inputs": [], "main_out": "init", "params": [{"name": "method", "type": "enum", "options": ["kaiming_normal_", "kaiming_uniform_", "xavier_normal_", "xavier_uniform_", "normal_", "uniform_", "zeros_", "ones_"], "default": "kaiming_normal_"}, {"name": "mean", "type": "float", "default": 0.0}, {"name": "std", "type": "float", "default": 1.0}], "description": "权重初始化。"}
    },

    "Testing (部署推理)": {
        "Inference Config": {"name": "Inference Config", "inputs": [], "main_out": "", "params": [{"name": "weights_path", "type": "string", "default": "./weights/model.pth"}, {"name": "device", "type": "enum", "options": ["cuda", "cpu"], "default": "cuda"}], "description": "推理配置。"}
    }
}