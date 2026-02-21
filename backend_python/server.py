import asyncio
import websockets
import json
import inspect
import torch.nn as nn
from shape_engine import infer_graph_shapes
from system_nodes import SYSTEM_NODES # 导入你的系统节点

def generate_torch_library():
    print("正在扫描 PyTorch 库...")
    # 把系统节点直接作为基础框架
    library = SYSTEM_NODES.copy()
    
    # 额外创建 PyTorch 分类
    library.update({
        "Convolutions (卷积层)": {}, "Linear (全连接层)": {}, 
        "Activations (激活函数)": {}, "Losses (损失函数)": {}, "Utilities (其他工具)": {}
    })

    # 动态遍历 PyTorch 库
    for name, obj in inspect.getmembers(nn):
        if inspect.isclass(obj) and issubclass(obj, nn.Module) and obj != nn.Module:
            category = "Utilities (其他工具)"
            if "Conv" in name: category = "Convolutions (卷积层)"
            elif "Linear" in name or "Bilinear" in name: category = "Linear (全连接层)"
            elif name in ["ReLU", "Sigmoid", "Tanh", "LeakyReLU", "Softmax", "GELU"]: category = "Activations (激活函数)"
            elif "Loss" in name: category = "Losses (损失函数)"

            doc_string = inspect.getdoc(obj)
            short_doc = "暂无说明文档"
            if doc_string:
                paragraphs = doc_string.split('\n\n')
                if paragraphs:
                    short_doc = paragraphs[0].replace('\n', ' ')
                    if len(short_doc) > 200: short_doc = short_doc[:197] + "..."

            params = []
            try:
                sig = inspect.signature(obj.__init__)
                for param_name, param in sig.parameters.items():
                    if param_name in ['self', 'args', 'kwargs', 'device', 'dtype']: continue
                    p_type = "string"
                    p_default = ""
                    if param.default != inspect.Parameter.empty:
                        if isinstance(param.default, bool): p_type = "bool"
                        elif isinstance(param.default, int): p_type = "int"
                        elif isinstance(param.default, float): p_type = "float"
                        elif isinstance(param.default, tuple): p_type = "tuple"
                        p_default = str(param.default) if param.default is not None else "None"
                    params.append({"name": param_name, "type": p_type, "default": p_default})
            except ValueError:
                pass

            # 统一成新架构，将 main_in 转换为 inputs 数组
            library[category][name] = {
                "name": name,
                "inputs": ["in"] if "Loss" not in name else ["preds", "targets"],
                "main_out": "out" if "Loss" not in name else "loss",
                "params": params,
                "description": short_doc
            }
    return {k: v for k, v in library.items() if v}

TORCH_NODE_LIBRARY = generate_torch_library()

async def handle_client(websocket):
    print("Godot 客户端已连接！")
    try:
        async for message in websocket:
            if message == "REQUEST_NODES":
                await websocket.send(json.dumps({"type": "LIBRARY", "data": TORCH_NODE_LIBRARY}))
            elif message.startswith("INFER_SHAPES:"):
                graph_json_str = message.replace("INFER_SHAPES:", "")
                shape_results = infer_graph_shapes(json.loads(graph_json_str))
                await websocket.send(json.dumps({"type": "SHAPE_RESULTS", "data": shape_results}))
            elif message.startswith("EXPORT_PY:"):
                req = json.loads(message.replace("EXPORT_PY:", ""))
                try:
                    from compiler import generate_pytorch_code
                    code = generate_pytorch_code(req["data"], req["class_name"])
                    # 直接由 Python 后端写入指定路径
                    with open(req["path"], "w", encoding="utf-8") as f:
                        f.write(code)
                    await websocket.send(json.dumps({"type": "EXPORT_SUCCESS", "msg": f"成功导出标准 PyTorch 模型到：\n{req['path']}"}))
                except Exception as e:
                    await websocket.send(json.dumps({"type": "EXPORT_ERROR", "msg": str(e)}))
    except websockets.exceptions.ConnectionClosed:
        print("Godot 客户端断开连接。")

async def main():
    print("启动 Python 服务器 (ws://localhost:8765)...")
    async with websockets.serve(handle_client, "localhost", 8765):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())