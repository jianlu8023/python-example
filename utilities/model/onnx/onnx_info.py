import onnx
import onnxruntime as ort

def check_onnx(onnx_path: str):
    """
    check_onnx 检查onnx格式rfdetr模型信息
    :param onnx_path:
    :return:
    """

    # --- 模型输入信息 ---
    # 输入名称: input
    # 预期形状: [1, 3, 576, 576]
    # 数据类型: tensor(float)
    # --------------------
    #
    # --- 模型输出信息 ---
    # 输出名称: dets
    # 输出形状: [1, 300, 4]
    # --------------------
    # 输出名称: labels
    # 输出形状: [1, 300, 9]
    # --------------------

    session = ort.InferenceSession(onnx_path)

    # 遍历所有输入节点
    print("--- 模型输入信息 ---")
    for input_node in session.get_inputs():
        print(f"输入名称: {input_node.name}")
        print(f"预期形状: {input_node.shape}")  # 重点看这里
        print(f"数据类型: {input_node.type}")
        print("-" * 20)

    # 顺便看看输出节点，方便后处理
    print("\n--- 模型输出信息 ---")
    for output_node in session.get_outputs():
        print(f"输出名称: {output_node.name}")
        print(f"输出形状: {output_node.shape}")
        print("-" * 20)


def get_shape_with_onnx(model_path):
    """
    get_shape_with_onnx
    :param model_path:
    :return:
    """
    # 节点名称: input, 形状: [1, 3, 576, 576]

    model = onnx.load(model_path)
    graph = model.graph

    for input_node in graph.input:
        # 获取维度信息
        shape = []
        for dim in input_node.type.tensor_type.shape.dim:
            # dim_value 是静态值，dim_param 是动态参数名
            if dim.HasField("dim_value"):
                shape.append(dim.dim_value)
            elif dim.HasField("dim_param"):
                shape.append(dim.dim_param)
            else:
                shape.append("?")

        print(f"节点名称: {input_node.name}, 形状: {shape}")
