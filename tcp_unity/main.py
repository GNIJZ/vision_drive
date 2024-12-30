import os.path
import sys
import time
import socket
import cv2

import onnxruntime as ort
import numpy as np
import mediapipe as mp



# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

            draw_face_landmarks(image, landmarks)  # 绘制面部关键点

            return np.array([eye_distance, eyebrow_distance, nose_eye_distance, mouth_distance, mouth_height, angle])
        else:
            return np.zeros(6)  # 如果没有检测到面部，返回零特征




BASE_DIR = os.path.dirname(os.path.realpath(sys.argv[0]))
print(BASE_DIR)

label_mapping = {'路怒': 0, '其他': 1}
reverse_label_mapping = {v: k for k, v in label_mapping.items()}
onnx_model_path = "emotion_two_11_7.onnx"
onnx_model_path = os.path.join(BASE_DIR, "emotion_two_11_7.onnx")
session = ort.InferenceSession(onnx_model_path,providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name  # 获取第一个输入的名称
pose_feature_name = session.get_inputs()[1].name   # 假设 `pose_feature` 是你模型中需要的第二个输入
face_detection = FaceDetection(0.5)
cap = cv2.VideoCapture(1)

host = '127.0.0.1'  # 本地地址
port = 7777         # 端口号，确保 Unity 监听的是相同的端口
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((host, port))

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
upper_body_indices = [ 9, 10, 11, 12,13,14,15,16]


def draw_upper_body_landmarks(image, pose_landmarks):
    for idx in upper_body_indices:
        landmark = pose_landmarks[idx]

        # 将相对坐标转换为像素坐标
        x = int(landmark.x * image.shape[1])  # 横坐标
        y = int(landmark.y * image.shape[0])  # 纵坐标

        # 确保绘制时坐标不会超出图像边界
        x = min(max(x, 0), image.shape[1] - 1)
        y = min(max(y, 0), image.shape[0] - 1)

        # 调试输出每个关键点的坐标
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            # 如果坐标有效，绘制关键点
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)  # 绘制绿色圆点
        else:
            # 如果坐标无效，则跳过绘制
            continue
        # 绘制上半身关键点



def get_face_landmarks(image,mp_face_mesh):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        return landmarks
    return None

def get_pose_landmarks(image,pose):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        return landmarks
    return None
def draw_face_landmarks(image, landmarks):
    for landmark in landmarks:
        x = int(landmark.x * image.shape[1])
        y = int(landmark.y * image.shape[0])
        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)  # 绿色圆点

def draw_pose_landmarks(image, landmarks):
    for landmark in landmarks:
        x = int(landmark.x * image.shape[1])
        y = int(landmark.y * image.shape[0])
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)  # 红色圆点
# 循环发送数据
try:
    while True:
        _, image = cap.read()

        pose_landmarks = get_pose_landmarks(image, pose)
        if pose_landmarks:
            draw_upper_body_landmarks(image, pose_landmarks)

        if face_detection.detector_result(image) is not None:
            (x, y, w, h) = face_detection.detector_result(image)
            cv2.rectangle(image, (x, y), (x + int(1.1*w), y + int(1.1*h)), (0, 255, 0), 2)
            face_fearture = face_detection.get_face_pose(image)

            face_fearture = face_fearture.astype(np.float32)  # 转换为 float32
            face_fearture = np.expand_dims(face_fearture, axis=0)  # 添加一个批次维度，变成 [1, 6]
            print(face_fearture)
            image_resized = cv2.resize(image, (64, 64))
            image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
            image_data = image_gray.astype(np.float32) / 255.0  # 转换为 float32
            image_data = np.expand_dims(image_data, axis=0)  # 增加 batch 维度 (shape: [1, 64, 64])
            image_data = np.expand_dims(image_data, axis=0)
            # tensor0 = torch.tensor(image_data).to(device=device)
            # tensor1 = torch.tensor(face_fearture).to(device=device)
            output = session.run(None, {
                input_name: image_data,
                pose_feature_name: face_fearture  # 添加姿势特征输入
            })
            predicted_class = np.argmax(output[0])  # 获取最大得分对应的索引，即预测的类别
            print(reverse_label_mapping[predicted_class])
            cv2.putText(image, f"rage: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
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


