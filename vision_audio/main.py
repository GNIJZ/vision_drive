from torch import nn
from torchvision import transforms
from data_set_0 import CustomDataset
import torch.utils.data

from model.model import EmotionNet

train_data_file = "dataset/data0/train.csv"
test_data_file = "dataset/data0/test.csv"

data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 使用显卡

train_image_folder = 'dataset/data0/train'
test_image_folder = 'dataset/data0/test'
train_custom_dataset = CustomDataset(train_data_file, train_image_folder, data_transform)
test_custom_dataset = CustomDataset(test_data_file, test_image_folder, data_transform)

# 修改模型的输出类别为2
EmotionNet = EmotionNet(2).to(device)

batch_size = 512

data_loader = torch.utils.data.DataLoader(dataset=train_custom_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_custom_dataset, batch_size=batch_size, shuffle=False)

Loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(EmotionNet.parameters(), lr=0.001)

num_epochs = 100
best_accuracy = 0.0
best_model_path = 'model/model_pretrain/Emotion_two_11_12.pth'

def evaluate_rage_model(model, data_loader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images,face_fearture,labels in data_loader:
            images = images.to(device)

            face_fearture=face_fearture.to(device)
            labels = labels.to(device)

            outputs = model(images,face_fearture)
            _, predicted = torch.max(outputs.data, 1)
            # 计算准确率
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    accuracy = 100 * correct / total if total > 0 else 0
    return accuracy

def evaluate_model(model, data_loader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            # 将标签转换为二分类（0：愤怒和厌恶，1：其他）
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # 计算准确率
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total if total > 0 else 0
    return accuracy

if __name__ == '__main__':
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images,face_fearture,labels) in enumerate(data_loader):
            EmotionNet.train()
            images = images.to(device)
            face_fearture=face_fearture.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outs = EmotionNet(images,face_fearture)
            loss = Loss_function(outs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(data_loader)}], Loss: {running_loss / 10:.4f}')
                running_loss = 0.0

        # 每10个epoch评估模型并保存最佳模型
        if (epoch + 1) % 10 == 0:
            accuracy = evaluate_rage_model(EmotionNet, test_loader)
            print(f'Accuracy after epoch {epoch + 1}: {accuracy:.2f}%')
            if accuracy > best_accuracy:
                best_accuracy = accuracy
            torch.save(EmotionNet.state_dict(), best_model_path)
