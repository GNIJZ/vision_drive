import torch
import torchvision.models as models
from mt.model_resnet import ResNet50,BasicBlock
import torch.onnx
import onnx
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


resnet50_gnij=ResNet50(BasicBlock,8).to(device)
checkpoint = torch.load('model/resnet_best_model.pth')

resnet50_gnij.load_state_dict(checkpoint)
resnet50_gnij.eval().to(device)
x=torch.randn(1, 3, 96, 96).to(device)
output=resnet50_gnij(x)

with torch.no_grad():
    torch.onnx.export(
        resnet50_gnij,
        x,
        'model/resnet50_gnij.onnx',
        opset_version=11,
        input_names=['input'],
        output_names=['output']
        )


#验证模型
onxx_model=onnx.load("model/resnet50_gnij.onnx")

onnx.checker.check_model(onxx_model)









