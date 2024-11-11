import time
import cv2
import numpy as np
import onnx
import onnxruntime as ort

# 加载 ONNX 模型
model_path = "mediapipe_face_detection.onnx"  # 替换为你的模型路径
model = onnx.load(model_path)
onnx.checker.check_model(model)

# 创建一个 ONNX Runtime session
session = ort.InferenceSession(model_path)

# 获取模型的输入名称
input_name = session.get_inputs()[0].name

# 使用 OpenCV 打开摄像头
cap = cv2.VideoCapture(0)  # 0 是默认摄像头设备ID

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    # 捕获视频帧
    ret, frame = cap.read()
    if not ret:
        print("无法读取视频帧")
        break

    # 转换为 RGB 图像并进行预处理
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 这里可以根据你的模型输入要求进行调整（例如，归一化、调整大小等）
    resized_image = cv2.resize(rgb_frame, (128, 128))  # 假设模型输入大小是 128x128
    image_data = resized_image.astype(np.float32) / 255.0  # 归一化
    image_data = np.transpose(image_data, (2, 0, 1))  # 转换为 (C, H, W)
    image_data = np.expand_dims(image_data, axis=0)  # 增加批次维度 (1, C, H, W)

    # 进行推理
    outputs = session.run(None, {input_name: image_data})

    # 假设第一个输出是边界框，第二个输出是置信度
    boxes = outputs[0]  # boxes 的形状为 (1, num_boxes, 4)
    confidences = outputs[1]  # confidences 的形状为 (1, num_boxes, 1)

    # 获取置信度数组并转换为一维
    confidence_values = confidences[0, :, 0]
    # 找到置信度最大的索引
    max_confidence_index = np.argmax(confidence_values)
    # 获取最大置信度对应的框
    max_confidence_box = boxes[0, max_confidence_index]  # 获取最大置信度对应的边界框 [x, y, w, h]
    # 提取坐标信息
    x, y, w, h = max_confidence_box
    # 转换为整数坐标
    x, y, w, h = map(int, [x, y, w, h])
    # 确保坐标没有超出图像边界
    x, y = max(0, x), max(0, y)
    x2, y2 = x + w, y + h

    # 绘制矩形框
    cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)  # 绿色框，线宽为2

    # 显示图像
    cv2.imshow("Face Detection", frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # 延迟 0.2 秒
    time.sleep(0.2)

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()
