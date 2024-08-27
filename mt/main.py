import pandas as pd
import torch
from pyrealsense2 import device
from torch import nn
from torchvision import transforms, datasets
from data_set_0 import CustomDataset
import torch.utils.data

from mt.model_resnet import ResNet50,BasicBlock

# 数据集目录


# 检查GPU是否可用



tranin_data_file="dataset/train.csv"
test_data_file="dataset/test.csv"

# data = pd.read_csv(tranin_data_file)
data_transform = transforms.Compose([
    # transforms.Resize((224, 224)),  # 调整图像大小为 224x224
    #转为灰度图
    # transforms.ToPILImage(),  # 将输入转换为 PIL 图像
    # transforms.Grayscale(num_output_channels=1),  # 转换为灰度图像，num_
    transforms.ToTensor()          # 转换为张量
])

# print(data)
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_image_folder = 'dataset/train'
test_image_folder='dataset/test'
train_custom_dataset = CustomDataset(tranin_data_file,train_image_folder,data_transform)

test_custom_dataset=CustomDataset(test_data_file,test_image_folder,data_transform)



resnet50=ResNet50(BasicBlock,8).to(device)

batch_size = 512

data_loader = torch.utils.data.DataLoader(dataset=train_custom_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_custom_dataset, batch_size=batch_size, shuffle=False)

Loss_function = nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(resnet50.parameters(),lr=0.001)

num_epochs = 100
#保存模型
best_accuracy = 0.0  # 初始化最佳准确率
best_model_path = 'model/resnet_best_model.pth'  # 指定保存最佳模型参数的路径


for epoch in range(num_epochs):
    running_loss =0.0
    for i, (images, labels) in enumerate(data_loader):
        resnet50.train()
        images = images.to(device)
        print("11111111111111",images.shape)
        labels = labels.to(device)
        optimizer.zero_grad()

        outs = resnet50(images)
        loss = Loss_function(outs,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i+1) % 10 == 0:  # 每100个batch打印一次
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(data_loader)}], Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

    if (epoch + 1) % 10 == 0:  # 每10个epoch保存一次最佳模型
        total=0
        correct=0
        resnet50.eval()
        with torch.no_grad():
            for j, (images, labels) in enumerate(test_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = resnet50(images)
                _,predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print("accuracy",accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(resnet50.state_dict(), best_model_path)

