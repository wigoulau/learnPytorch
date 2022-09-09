import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 定义超参数
epochs = 3
BATCH_SIZE_TRAIN = 64   # 每次训练图像数
BATCH_SIZE_TEST = 100   # 每次测试图像数
LEARNING_RATE = 0.01    # 学习率
MOMENTUM = 0.5
RANDON_SEED = 1         # 随机种子

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST("./data", train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
                               ])),
    batch_size=BATCH_SIZE_TRAIN, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST("./data", train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
                               ])),
    batch_size=BATCH_SIZE_TEST, shuffle=True
)

examples = enumerate(train_loader)
print(examples)
idx, (data, target) = next(examples)
print(idx, target)
idx, (data, target) = next(examples)
print(idx, target)

fig = plt.figure()
pic_num = 4
for i in range(pic_num):
    plt.subplot(2, int(pic_num / 2), i + 1)
    plt.tight_layout()
    plt.imshow(data[i][0], cmap="gray", interpolation="none")
    plt.title("Groud truth:{}".format(target[i]))
    plt.xticks([])
    plt.yticks([])
#plt.show()

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

network = CNNNet().to(device)
optimizer = optim.SGD(network.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

LOG_INTERVAL = 10

def train(epoch):
    network.train()
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = network(data)
        #pred = output.data.max(1, keepdim=True)[1]
        #print("pred is {}, target is {}".format(pred[0], target[0]))
        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, idx * len(data),
                                                                           len(train_loader.dataset),
                                                                           100. * idx / len(train_loader),
                                                                           loss.item()))
            torch.save(network.state_dict(), "./model.pth")
            torch.save(optimizer.state_dict(), "./optimizer.pth")

def test():
    network.eval()
    with torch.no_grad():
        correct = 0
        test_loss = 0
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = network(data)
            #print(idx, output[0], target[0])
            test_loss += torch.nn.functional.nll_loss(output, target, reduction="sum").item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


train(1)
train(2)
train(3)
train(4)
train(5)
test()
