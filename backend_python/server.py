import asyncio
import websockets
import json
import inspect
import torch.nn as nn
import torch.optim as optim
from shape_engine import infer_graph_shapes
from system_nodes import SYSTEM_NODES

def generate_torch_library():
    print("正在扫描 PyTorch 库...")
    library = SYSTEM_NODES.copy()
    
    # 初始化 PyTorch 专属分类
    library.update({
        "Convolutions (卷积层)": {}, "Linear (全连接层)": {}, 
        "Activations (激活函数)": {}, "Losses (损失函数)": {}, 
        "Optimizers (优化器)": {}, "Utilities (其他工具)": {}
    })

    # ==========================================
    # 1. 扫描网络层与 Loss 函数 (torch.nn)
    # ==========================================
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
                short_doc = doc_string.split('\n\n')[0].replace('\n', ' ')[:197]

            params = []
            try:
                sig = inspect.signature(obj.__init__)
                for p_name, param in sig.parameters.items():
                    if p_name in ['self', 'args', 'kwargs', 'device', 'dtype']: continue
                    p_type, p_default = "string", ""
                    if param.default != inspect.Parameter.empty:
                        if isinstance(param.default, bool): p_type = "bool"
                        elif isinstance(param.default, int): p_type = "int"
                        elif isinstance(param.default, float): p_type = "float"
                        elif isinstance(param.default, tuple): p_type = "tuple"
                        p_default = str(param.default) if param.default is not None else "None"
                    params.append({"name": p_name, "type": p_type, "default": p_default})
            except ValueError: pass

            # 如果是 Loss 函数，强制设置为双输入 (preds, targets)
            is_loss = (category == "Losses (损失函数)")
            library[category][name] = {
                "name": name,
                "inputs": ["preds", "targets"] if is_loss else ["in"],
                "main_out": "loss" if is_loss else "out",
                "params": params,
                "description": short_doc
            }

    # ==========================================
    # 2. 扫描优化器 (torch.optim)
    # ==========================================
    for name, obj in inspect.getmembers(optim):
        if inspect.isclass(obj) and issubclass(obj, optim.Optimizer) and obj != optim.Optimizer:
            params = []
            try:
                sig = inspect.signature(obj.__init__)
                for p_name, param in sig.parameters.items():
                    if p_name in ['self', 'params']: continue 
                    p_type, p_default = "string", ""
                    if param.default != inspect.Parameter.empty:
                        if isinstance(param.default, bool): p_type = "bool"
                        elif isinstance(param.default, int): p_type = "int"
                        elif isinstance(param.default, float): p_type = "float"
                        p_default = str(param.default)
                    params.append({"name": p_name, "type": p_type, "default": p_default})
            except ValueError: pass
            
            library["Optimizers (优化器)"][name] = {
                "name": name,
                "inputs": ["loss"], 
                "main_out": "step",
                "params": params,
                "description": "PyTorch 优化器"
            }

    # 剔除空分类并返回
    return {k: v for k, v in library.items() if v}

TORCH_NODE_LIBRARY = generate_torch_library()

# ==========================================
# 核心 WebSocket 通讯与指令分发中心
# ==========================================
async def handle_client(websocket):
    print("Godot 客户端已连接！")
    try:
        async for message in websocket:
            
            # 1. 发送节点库数据
            if message == "REQUEST_NODES":
                await websocket.send(json.dumps({"type": "LIBRARY", "data": TORCH_NODE_LIBRARY}))
                
            # 2. 实时推导张量形状
            elif message.startswith("INFER_SHAPES:"):
                graph_json_str = message.replace("INFER_SHAPES:", "")
                shape_results = infer_graph_shapes(json.loads(graph_json_str))
                await websocket.send(json.dumps({"type": "SHAPE_RESULTS", "data": shape_results}))
                
            # 3. 导出模型专属代码 (.py) [来自 model_editor]
            elif message.startswith("EXPORT_PY:"):
                req = json.loads(message.replace("EXPORT_PY:", ""))
                try:
                    from compiler import generate_pytorch_code
                    code = generate_pytorch_code(req["data"], req["class_name"])
                    with open(req["path"], "w", encoding="utf-8") as f:
                        f.write(code)
                    await websocket.send(json.dumps({"type": "EXPORT_SUCCESS", "msg": f"成功导出标准 PyTorch 模型代码到：\n{req['path']}"}))
                except Exception as e:
                    await websocket.send(json.dumps({"type": "EXPORT_ERROR", "msg": str(e)}))
                    
            # 4. 导出训练闭环脚本代码 (train.py) [来自 train_editor]
            elif message.startswith("EXPORT_TRAIN_PY:"):
                req = json.loads(message.replace("EXPORT_TRAIN_PY:", ""))
                try:
                    from compiler import generate_train_code
                    code = generate_train_code(req["data"], req["class_name"])
                    with open(req["path"], "w", encoding="utf-8") as f:
                        f.write(code)
                    await websocket.send(json.dumps({"type": "EXPORT_SUCCESS", "msg": f"成功导出完整训练脚本到：\n{req['path']}"}))
                except Exception as e:
                    await websocket.send(json.dumps({"type": "EXPORT_ERROR", "msg": str(e)}))
                    
            # 5. 导出推理部署封装类 (test.py) [来自 test_editor]
            elif message.startswith("EXPORT_TEST_PY:"):
                req = json.loads(message.replace("EXPORT_TEST_PY:", ""))
                try:
                    from compiler import generate_test_code
                    code = generate_test_code(req["data"], req["class_name"])
                    with open(req["path"], "w", encoding="utf-8") as f:
                        f.write(code)
                    await websocket.send(json.dumps({"type": "EXPORT_SUCCESS", "msg": f"成功导出推理部署类到：\n{req['path']}"}))
                except Exception as e:
                    await websocket.send(json.dumps({"type": "EXPORT_ERROR", "msg": str(e)}))
                    
    except websockets.exceptions.ConnectionClosed:
        print("Godot 客户端断开连接。")

async def main():
    print("启动 Python 服务器 (ws://127.0.0.1:8765)...")
    # 强制绑定本地 IPv4，防止 Godot 连接失败
    async with websockets.serve(handle_client, "127.0.0.1", 8765):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())