import os.path
import sys
import time
import socket
import cv2
import numpy as np
import torch
import onnxruntime as ort
from eval import FaceDetection

BASE_DIR = os.path.dirname(os.path.realpath(sys.argv[0]))
print(BASE_DIR)

label_mapping = {'路怒': 0, '其他': 1}
reverse_label_mapping = {v: k for k, v in label_mapping.items()}
onnx_model_path = "emotion_two_11_7.onnx"
onnx_model_path = os.path.join(BASE_DIR, "emotion_two_11_7.onnx")
session = ort.InferenceSession(onnx_model_path)
input_name = session.get_inputs()[0].name  # 获取第一个输入的名称
pose_feature_name = session.get_inputs()[1].name   # 假设 `pose_feature` 是你模型中需要的第二个输入
face_detection = FaceDetection(0.5)
cap = cv2.VideoCapture(0)

host = '127.0.0.1'  # 本地地址
port = 7777         # 端口号，确保 Unity 监听的是相同的端口
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((host, port))

# 循环发送数据
try:
    while True:
        _, image = cap.read()
        if face_detection.detector_result(image) is not None:
            (x, y, w, h) = face_detection.detector_result(image)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_fearture = face_detection.get_face_pose(image)
            face_fearture = face_fearture.astype(np.float32)  # 转换为 float32
            face_fearture = np.expand_dims(face_fearture, axis=0)  # 添加一个批次维度，变成 [1, 6]
            image_resized = cv2.resize(image, (64, 64))
            image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
            image_data = image_gray.astype(np.float32) / 255.0  # 转换为 float32
            image_data = np.expand_dims(image_data, axis=0)  # 增加 batch 维度 (shape: [1, 64, 64])
            image_data = np.expand_dims(image_data, axis=0)
            tensor0 = torch.tensor(image_data).to(device='cuda:0')
            tensor1 = torch.tensor(face_fearture).to(device='cuda:0')
            output = session.run(None, {
                input_name: image_data,
                pose_feature_name: face_fearture  # 添加姿势特征输入
            })
            predicted_class = np.argmax(output[0])  # 获取最大得分对应的索引，即预测的类别
            print(reverse_label_mapping[predicted_class])
            cv2.putText(image, f"路怒状态: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            #client_socket.sendall(predicted_class)  # 将字符串转换为字节并发送
            client_socket.send(str(predicted_class).encode('utf-8'))
            time.sleep(0.1)
        cv2.imshow('Camera Feed', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Client stopped.")
finally:
    cap.release()
    cv2.destroyAllWindows()
    client_socket.close()  # 关闭连接


