# -*- coding: utf-8 -*-

import time
import cv2
import mediapipe as mp
import numpy as np
import torch
from torchvision.transforms import transforms
import onnxruntime as ort

from vision_emotion.model.model import EmotionNet

# label_mapping = {
#     '路怒': [0, 1],  # 路怒对应类别 0 和 1
#     '其他': [2, 3, 4, 5, 6]  # 其他对应类别 2, 3, 4, 5, 6
# }
# label_mapping = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}
label_mapping = {'路怒': 0, '其他': 1}
reverse_label_mapping = {v: k for k, v in label_mapping.items()}
# reverse_label_mapping = {v: k for k, values in label_mapping.items() for v in values}
onnx_model_path = "model/onxx/emotion_two_11_7.onnx"
session = ort.InferenceSession(onnx_model_path)
input_name = session.get_inputs()[0].name  # 获取第一个输入的名称
pose_feature_name = session.get_inputs()[1].name   # 假设 `pose_feature` 是你模型中需要的第二个输入

# best_model_path= 'model/model_pretrain/Emotion_two_11_7.pth'

# best_model = EmotionNet(2).to(device="cuda:0")  # 创建模型实例
# best_model.load_state_dict(torch.load(best_model_path,map_location='cuda:0'))  # 加载最佳参数
# best_model.eval()  # 切换到评估模式
class FaceDetection:
    def __init__(self, min_detection_confidence=0.5, margin=20):
        self.face_detector = mp.solutions.face_detection.FaceDetection(model_selection=1,
                                                                       min_detection_confidence=min_detection_confidence)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5)
        self.margin = margin

    def detector_result(self, image):
        results = self.face_detector.process(image)
        if results.detections is None or len(results.detections) != 1:
            return None

        bbox = results.detections[0].location_data.relative_bounding_box
        x = bbox.xmin * image.shape[1]
        y = bbox.ymin * image.shape[0]
        w = bbox.width * image.shape[1]
        h = bbox.height * image.shape[0]

        # 增加边距
        x = round(x) - self.margin
        y = round(y) - self.margin
        w = round(w) + 2 * self.margin
        h = round(h) + 2 * self.margin

        # 边界限制
        x = min(max(x, 0), image.shape[1])
        y = min(max(y, 0), image.shape[0])
        w = min(max(w, 0), image.shape[1] - x)
        h = min(max(h, 0), image.shape[0] - y)

        face_image = image[y:y + h, x:x + w]
        return (x, y, w, h)


    def get_face_pose(self, image):
        # 使用 MediaPipe 提取面部关键点
        results = self.face_mesh.process(image)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            # 提取关键点之间的距离和角度作为特征
            left_eye = landmarks[33]
            right_eye = landmarks[133]
            eye_distance = np.sqrt((left_eye.x - right_eye.x) ** 2 + (left_eye.y - right_eye.y) ** 2)

            left_eyebrow = landmarks[17]
            right_eyebrow = landmarks[22]
            eyebrow_distance = np.sqrt(
                (left_eyebrow.x - right_eyebrow.x) ** 2 + (left_eyebrow.y - right_eyebrow.y) ** 2)

            left_eye = landmarks[33]
            nose = landmarks[1]
            nose_eye_distance = np.sqrt((nose.x - left_eye.x) ** 2 + (nose.y - left_eye.y) ** 2)

            left_mouth = landmarks[48]
            right_mouth = landmarks[54]
            mouth_distance = np.sqrt((left_mouth.x - right_mouth.x) ** 2 + (left_mouth.y - right_mouth.y) ** 2)

            top_mouth = landmarks[62]
            bottom_mouth = landmarks[66]
            mouth_height = np.sqrt((top_mouth.x - bottom_mouth.x) ** 2 + (top_mouth.y - bottom_mouth.y) ** 2)

            # 计算眼睛之间的角度
            left_eye = landmarks[33]
            right_eye = landmarks[133]
            eye_distance_x = right_eye.x - left_eye.x
            eye_distance_y = right_eye.y - left_eye.y
            angle = np.arctan2(eye_distance_y, eye_distance_x)

            return np.array([eye_distance, eyebrow_distance, nose_eye_distance, mouth_distance, mouth_height, angle])
        else:
            return np.zeros(6)  # 如果没有检测到面部，返回零特征

if __name__ == '__main__':
    face_detection = FaceDetection(0.5)

    cap = cv2.VideoCapture(0)
    while True:
        _, image = cap.read()

        if face_detection.detector_result(image) is not None:
            (x, y, w, h) = face_detection.detector_result(image)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            face_fearture=face_detection.get_face_pose(image)
            face_fearture = face_fearture.astype(np.float32)  # 转换为 float32
            face_fearture = np.expand_dims(face_fearture, axis=0)  # 添加一个批次维度，变成 [1, 6]

            image_resized = cv2.resize(image, (64, 64))
            # 3. 将图像转换为灰度图
            image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
            image_data = image_gray.astype(np.float32) / 255.0  # 转换为 float32
            image_data = np.expand_dims(image_data, axis=0)  # 增加 batch 维度 (shape: [1, 64, 64])
            image_data = np.expand_dims(image_data, axis=0)
            # 6. 如果需要将其转换为 4D 张量（批次大小，通道数，高度，宽度）
            # image_data = image_data.unsqueeze(0).unsqueeze(0)

            tensor0 = torch.tensor(image_data).to(device='cuda:0')
            tensor1=torch.tensor(face_fearture).to(device='cuda:0')
            output = session.run(None, {
                input_name: image_data,
                pose_feature_name: face_fearture  # 添加姿势特征输入
            })
            #output=best_model(tensor0,tensor1)

            # predicted_class = torch.argmax(output[0]).item()
            predicted_class = np.argmax(output[0])  # 获取最大得分对应的索引，即预测的类别
            print(output)
            print(reverse_label_mapping[predicted_class])
            time.sleep(0.1)

        cv2.imshow('Camera Feed', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()