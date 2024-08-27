import time

import cv2
import mediapipe as mp

class FaceDetection:
    def __init__(self, min_detection_confidence=0.5):
        self.face_detector = mp.solutions.face_detection.FaceDetection(model_selection=0,
                                                                       min_detection_confidence=min_detection_confidence)

    def detector_result(self, image):
        results = self.face_detector.process(image)
        if results.detections is None or len(results.detections) != 1:
            return None

        bbox = results.detections[0].location_data.relative_bounding_box
        x = bbox.xmin * image.shape[1]
        y = bbox.ymin * image.shape[0]
        w = bbox.width * image.shape[1]
        h = bbox.height * image.shape[0]

        # 将计算转移到外部，只进行一次 round()、min()、max() 和图像切片操作
        x, y, w, h = round(x), round(y), round(w), round(h)
        x = min(max(x, 0), image.shape[1])
        y = min(max(y, 0), image.shape[0])
        w = min(max(w, 0), image.shape[1] - x)
        h = min(max(h, 0), image.shape[0] - y)

        face_image = image[y:y + h, x:x + w]
        return (x, y, w, h)


if __name__ == '__main__':
    face_detection = FaceDetection(0.5)

    cap = cv2.VideoCapture(0)
    while True:
        _, image = cap.read()

        if face_detection.detector_result(image) is not None:
            (x, y, w, h) = face_detection.detector_result(image)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Camera Feed', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
