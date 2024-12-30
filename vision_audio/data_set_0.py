import os

import numpy as np
import pandas as pd
from PIL import Image
import mediapipe as mp
import torch.utils.data
from matplotlib import pyplot as plt
from torchvision import transforms
import librosa

#label_mapping = {'anger': 0, 'contempt': 1, 'disgust': 2, 'fear': 3, 'happy': 4, 'neutral':5,'sad':6, 'surprise': 7}
#路怒标签
label_mapping = {'angry': 0, 'disgust': 0, 'contempt': 1, 'fear': 1, 'happy': 1, 'neutral': 1, 'sad': 1, 'surprise': 1}

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,data_file,image_folder,transform=None):
        super(CustomDataset, self).__init__()
        self.data = pd.read_csv(data_file)  # 从 CSV 文件中读取数据
        self.image_folder = image_folder  # 图像文件夹路径
        self.transform = transform  # 数据预处理操
        self.label_mapping = label_mapping

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh()

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
            return np.array([eye_distance, eyebrow_distance,nose_eye_distance,mouth_distance,mouth_height,angle])
        else:
            return np.zeros(6)  # 如果没有检测到面部，返回零特征
#得到每一张图片标签和数据，还要转tensor
    def __getitem__(self, index):
        label = self.data.iloc[index,0]
        filename = self.data.iloc[index, 1]
        image_path = self.image_folder + '/' + label
        image = Image.open(os.path.join(image_path,filename))
        pose_features = torch.tensor(self.get_face_pose(image))
        pose_features = pose_features.float()
        if self.transform:
            image = self.transform(image)
        labelmap = self.label_mapping[label]
        # plt.imshow(image.squeeze(0), cmap='gray')  # 对于单通道图像使用 cmap='gray'
        # plt.title(f"Label: {label}")
        # plt.axis('off')
        # plt.show()
        return image,pose_features, labelmap
    def __len__(self):
        # 返回数据集大小
        return len(self.data)



# custom_dataset = CustomDataset(data_file,image_folder,data_transform)
# #
# #
# batch_size = 1
# data_loader = torch.utils.data.DataLoader(dataset=custom_dataset, batch_size=batch_size, shuffle=True)
# image_path='dataset/data0/train/angry'
# # for imagefile in os.listdir(image_path):
# #     image = Image.open(os.path.join(image_path,imagefile))
# #     resized_image =image.resize((224,224))
# #     print(resized_image.size)
#
# for images,face,labels in data_loader:
#     print(images.shape)

audio_all='data/audio/angry/'
def max_audio():
    max=16000
    for audio in os.listdir(audio_all):
        audio_data, sr = librosa.load(os.path.join(audio_all, audio),sr=None)
        if len(audio_data)>max:
            max=len(audio_data)
    return max

def mfcc_features(audio_data,maxlengh):
    if len(audio_data) < maxlengh:
        # 如果音频数据长度小于期望长度，使用0填充
        audio_data = np.pad(audio_data, (0, maxlengh - len(audio_data)), mode='constant')
    mfccs = librosa.feature.mfcc(y=audio_data, sr=16000, n_mfcc=13)  # 13维MFCC特征
    mfccs = mfccs.T
    return mfccs
maxlengh=max_audio()
for audio in os.listdir(audio_all):
    audio_data, sr = librosa.load(os.path.join(audio_all, audio), sr=None)
    data=mfcc_features(audio_data,maxlengh)
    print(data.shape)
# print(audio_data.shape)