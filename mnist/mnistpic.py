import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

random_seed = 1
torch.manual_seed(random_seed)
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        # 池化层
        self.pool = nn.MaxPool2d(2)

        # 激活函数
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d()

        # 全连接层
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.relu(x)
        x = self.relu(self.pool(self.dropout(self.conv2(x))))
        x = x.view(-1, 320)
        x = self.relu(self.fc1(x))
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.fc2(x)
        #print(x[0], max(x[0]))
        return torch.nn.functional.log_softmax(x, dim=1)


network = CNNNet()
network_state_dict = torch.load("model.pth")
network.load_state_dict(network_state_dict)

from PIL import Image
imgPath = "test.png"
img = Image.open(imgPath)
#img.show()
print(img)
#img = img.resize((28, 28), Image.ANTIALIAS)
img = img.convert("1")
print(img, img.size)
import PIL.ImageOps
#img = PIL.ImageOps.invert(img)
#img.show()
# fig = plt.figure()
# plt.tight_layout()
# plt.imshow(img, cmap='gray', interpolation='none')
# plt.title("Ground Truth: {}".format(3))
# plt.xticks([])
# plt.yticks([])
# plt.show()
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((28, 28)),
    torchvision.transforms.ToTensor()
]
)
img = transform(img)
print(img.shape)
#print(img)


output = network(img)
pred = output.data.max(1, keepdim=True)[1]
print("pred is {}".format(pred[0][0]))

fig = plt.figure()
plt.tight_layout()
plt.imshow(img.squeeze(), cmap='gray', interpolation='none')
plt.title("Ground Truth: {}".format(pred[0][0]))
plt.xticks([])
plt.yticks([])
plt.show()