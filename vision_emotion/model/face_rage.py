import mediapipe as mp
import cv2
import numpy as np

# 初始化MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)


# 提取面部关键点和计算比例
def get_face_pose(self, image):
    # 使用 MediaPipe 提取面部关键点
    # 转换为 RGB 图像，MediaPipe 使用的是 RGB 格式
    image_rgb = np.array(image)[:, :, ::-1]
    results = self.face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        # 获取第一个人脸的关键点
        landmarks = results.multi_face_landmarks[0].landmark
        # 提取你需要的姿态特征（例如，计算关键点间的角度等）
        # 假设你选择了简单的比例特征，比如两个关键点之间的距离
        left_eye = landmarks[33]
        right_eye = landmarks[133]
        # 计算眼睛间距的欧几里得距离
        eye_distance = np.sqrt((left_eye.x - right_eye.x) ** 2 + (left_eye.y - right_eye.y) ** 2)
        left_eyebrow = landmarks[17]
        right_eyebrow = landmarks[22]
        # 计算眉毛间距的欧几里得距离
        eyebrow_distance = np.sqrt(
            (left_eyebrow.x - right_eyebrow.x) ** 2 + (left_eyebrow.y - right_eyebrow.y) ** 2)
        left_eye = landmarks[33]
        nose = landmarks[1]
        # 计算眼睛到鼻子的距离
        nose_eye_distance = np.sqrt((nose.x - left_eye.x) ** 2 + (nose.y - left_eye.y) ** 2)
        left_mouth = landmarks[48]
        right_mouth = landmarks[54]
        # 计算嘴巴两侧的水平距离
        mouth_distance = np.sqrt((left_mouth.x - right_mouth.x) ** 2 + (left_mouth.y - right_mouth.y) ** 2)
        top_mouth = landmarks[62]
        bottom_mouth = landmarks[66]
        # 计算嘴巴上下的垂直距离
        mouth_height = np.sqrt((top_mouth.x - bottom_mouth.x) ** 2 + (top_mouth.y - bottom_mouth.y) ** 2)
        left_eye = landmarks[33]
        right_eye = landmarks[133]
        # 计算眼睛之间的水平距离
        eye_distance_x = right_eye.x - left_eye.x
        eye_distance_y = right_eye.y - left_eye.y
        # 计算眼睛之间的角度（弧度制）
        angle = np.arctan2(eye_distance_y, eye_distance_x)
        # 返回计算的特征
        return np.array([eye_distance, eyebrow_distance, nose_eye_distance, mouth_distance, mouth_height, angle])
    else:
        return np.zeros(6)  # 如果没有检测到面部，返回零特征


# 归一化面部比例（例如，将所有比例归一化到 [0, 1] 范围内）



# 读取图像
image = cv2.imread('face_image.jpg')

# 使用MediaPipe检测面部关键点
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = face_mesh.process(image_rgb)


