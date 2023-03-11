import torch
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
input_ = torch.rand(1, 3, 64, 64)



vgg16_model=models.vgg16(pretrained=True)
print(vgg16_model.classifier[6].out_features)
#vgg16 = models.vgg16_bn(num_classes = 11)
# vgg16_model.classifier=vgg16_model.classifier[:-1]
modules_vgg=list(vgg16_model.classifier[:-1])
vgg16_model=nn.Sequential(*modules_vgg)
output = vgg16_model(input_)
print(output.shape)


image = Image.open(r"C:\Users\user\Desktop\TorchSSL-main\dog.jpg")

# Get features part of the network
model = models.vgg16(pretrained=True)
tensor = transforms.ToTensor()(transforms.Resize((224, 224))(image)).unsqueeze(dim=0)

print(vgg16_model(tensor).shape)
model = models.vgg16(pretrained=True)
x = model.avgpool(tensor)
x = torch.flatten(x, 1)
final_x = model.classifier[0](x) # only first classifier layer

vgg16_model = models.vgg16(pretrained=True)
vgg16_model.classifier = vgg16_model.classifier[:-1] 

mdl = nn.Sequential(vgg16_model.features[::], vgg16_model.classifier[:-1])
print(mdl(tensor).shape)

x = model.features(tensor)
print(x.shape)
x = model.avgpool(x)
print(x.shape)
x = torch.flatten(x,1)
print(x.shape)
x = model.classifier[0](x)
print(x.shape)