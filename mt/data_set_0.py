import os

import pandas as pd
from PIL import Image

import torch.utils.data

# data_file="dataset/train.csv"
# data = pd.read_csv(data_file)
# print(data)

# image_folder = 'dataset/train'
#
# data_transform = transforms.Compose([
#     # transforms.Resize((224, 224)),  # 调整图像大小为 224x224
#     transforms.ToTensor(),           # 转换为张量
# ])

label_mapping = {'anger': 0, 'contempt': 1, 'disgust': 2, 'fear': 3, 'happy': 4, 'neutral':5,'sad':6, 'surprise': 7}
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,data_file,image_folder,transform=None):
        super(CustomDataset, self).__init__()
        self.data = pd.read_csv(data_file)  # 从 CSV 文件中读取数据
        self.image_folder = image_folder  # 图像文件夹路径
        self.transform = transform  # 数据预处理操
        self.label_mapping = label_mapping

#得到每一张图片标签和数据，还要转tensor
    def __getitem__(self, index):
        label = self.data.iloc[index,0]
        filename = self.data.iloc[index, 1]
        image_path = self.image_folder + '/' + label
        image = Image.open(os.path.join(image_path,filename))
        if self.transform:
            image = self.transform(image)
        labelmap = self.label_mapping[label]
        return image, labelmap
    def __len__(self):
        # 返回数据集大小
        return len(self.data)


# custom_dataset = CustomDataset(data_file,image_folder,data_transform)
#
#
# batch_size = 1
# data_loader = torch.utils.data.DataLoader(dataset=custom_dataset, batch_size=batch_size, shuffle=True)




# for images, labels in data_loader:
#     print(images.shape)