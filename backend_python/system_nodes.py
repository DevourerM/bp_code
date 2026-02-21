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