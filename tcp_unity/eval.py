import numpy as np
import mediapipe as mp


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