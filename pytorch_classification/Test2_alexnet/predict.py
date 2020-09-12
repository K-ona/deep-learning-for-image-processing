import torch
from model import AlexNet
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json

data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load image
img = Image.open(r'D:\Document\GitHub\deep-learning-for-image-processing\pytorch_classification\Test2_alexnet\test.jpg')
plt.imshow(img)
# [N, C, H, W]
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)

# read class_indict
try:
    json_file = open('class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# create model
model = AlexNet(num_classes=5)
# load model weights
model_weight_path = r"D:\Document\GitHub\deep-learning-for-image-processing\pytorch_classification\Test2_alexnet\AlexNet.pth"
model.load_state_dict(torch.load(model_weight_path))
model.eval()

with torch.no_grad():
    # predict class
    output = torch.squeeze(model(img))
    predict = torch.softmax(output, dim=0)
    print(predict)
    # predict_cla = torch.argmax(predict).numpy()
print(list((class_indict[str(predict_cla)], predict[predict_cla].item()) for predict_cla in range(5)))
plt.show()
