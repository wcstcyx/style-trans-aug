import torch
import torchvision
import torchvision.models
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
style_test = 0.0
style_val = 0.0
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(28),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 ]),
    "test": transforms.Compose([transforms.Resize((28, 28)),  # cannot 224, must (224, 224)
                               transforms.ToTensor(),
                              ]),
    "testdev": transforms.Compose([transforms.Resize((28, 28)),  # cannot 224, must (224, 224)
                               transforms.ToTensor(),
                              ]),
    "val": transforms.Compose([transforms.Resize((28, 28)),  # cannot 224, must (224, 224)
                               transforms.ToTensor(),
                              ])}
train_data = torchvision.datasets.ImageFolder(root = "/kaggle/input/style/style/train" ,   transform = data_transform["train"])
 
traindata = DataLoader(dataset=train_data, batch_size=128, shuffle=True, num_workers=0)  # 将训练数据以每次32张图片的形式抽出进行训练
 
test_data = torchvision.datasets.ImageFolder(root = "/kaggle/input/style/style/test-m" , transform = data_transform["test"])
testdev_data = torchvision.datasets.ImageFolder(root = "/kaggle/input/style/style/test" , transform = data_transform["testdev"])

val_data = torchvision.datasets.ImageFolder(root = "/kaggle/input/style/style/val" , transform = data_transform["val"])

train_size = len(train_data)  # 训练集的长度
test_size = len(test_data)
testdev_size = len(testdev_data)
val_size = len(val_data)  # 测试集的长度
print(train_size)   #输出训练集长度看一下，相当于看看有几张图片
print(test_size) 
print(testdev_size)#输出测试集长度看一下，相当于看看有几张图片
print(val_size)
testdata = DataLoader(dataset=test_data, batch_size=128, shuffle=True, num_workers=0)  
testdevdata = DataLoader(dataset=testdev_data, batch_size=128, shuffle=True, num_workers=0)# 将训练数据以每次32张图片的形式抽出进行测试
valdata = DataLoader(dataset=val_data, batch_size=128, shuffle=True, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("using {} device.".format(device))
 
 
class BasicBlock(nn.Module):
    expansion = 1
 
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
 
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
 
        out += identity
        out = self.relu(out)
 
        return out
 
 
class Bottleneck(nn.Module):
  
    expansion = 4
 
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, width_per_group=64):
                
        super(Bottleneck, self).__init__()
 
        width = int(out_channel * (width_per_group / 64.)) * groups
 
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
 
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
 
        out = self.conv3(out)
        out = self.bn3(out)
 
        out += identity
        out = self.relu(out)
 
        return out
 
 
class ResNet(nn.Module):
 
    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=10,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
 
        self.groups = groups
        self.width_per_group = width_per_group
 
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
                              
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)
 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
 
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
 
        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion
 
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))
 
        return nn.Sequential(*layers)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
 
        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
 
        return x
 
 
def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
 
 
def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
 
 
def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)
 
 
def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group) 
def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)
net = resnet34()
# load pretrain weights
# download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
model_weight_path = "/kaggle/input/style-model/style_model.pth"
assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
net.load_state_dict(torch.load(model_weight_path, map_location=device))

net.to(device)
print(net.to(device))  #输出模型结构
test1 = torch.ones(64, 3, 120, 120)  # 测试一下输出的形状大小 输入一个64,3,120,120的向量
test1 = net(test1.to(device))    #将向量打入神经网络进行测试
print(test1.shape)  #查看输出的结果
 
epoch = 200 # 迭代次数即训练次数
learning = 0.001  # 学习率
optimizer = torch.optim.Adam(net.parameters(), lr=learning)  # 使用Adam优化器-写论文的话可以具体查一下这个优化器的原理
loss = nn.CrossEntropyLoss()  # 损失计算方式，交叉熵损失函数
 
train_loss_all = []  # 存放训练集损失的数组
train_accur_all = []  # 存放训练集准确率的数组
test_loss_all = []  # 存放测试集损失的数组
test_accur_all = [] 
testdev_loss_all = []  # 存放测试集损失的数组
testdev_accur_all = [] 
val_loss_all = []  # 存放测试集损失的数组
val_accur_all = []# 存放测试集准确率的数组
for i in range(epoch):  #开始迭代
    train_loss = 0   #训练集的损失初始设为0
    train_num = 0.0   #
    train_accuracy = 0.0  #训练集的准确率初始设为0
    net.train()   #将模型设置成 训练模式
    train_bar = tqdm(traindata)  #用于进度条显示，没啥实际用处
    for step, data in enumerate(train_bar):  #开始迭代跑， enumerate这个函数不懂可以查查，将训练集分为 data是序号，data是数据
        img, target = data    #将data 分位 img图片，target标签
        optimizer.zero_grad()  # 清空历史梯度
        outputs = net(img.to(device))  # 将图片打入网络进行训练,outputs是输出的结果
        loss1 = loss(outputs, target.to(device))  # 计算神经网络输出的结果outputs与图片真实标签target的差别-这就是我们通常情况下称为的损失
        outputs = torch.argmax(outputs, 1)   #会输出10个值，最大的值就是我们预测的结果 求最大值
        loss1.backward()   #神经网络反向传播
        optimizer.step()  #梯度优化 用上面的abam优化
        train_loss += abs(loss1.item()) * img.size(0)  #将所有损失的绝对值加起来
        accuracy = torch.sum(outputs == target.to(device))   #outputs == target的 即使预测正确的，统计预测正确的个数,从而计算准确率
        train_accuracy = train_accuracy + accuracy   #求训练集的准确率
        train_num += img.size(0)  #
    print("epoch：{} ， train-Loss：{} , train-accuracy：{}".format(i + 1, train_loss / train_num, train_accuracy / train_num))
    train_loss_all.append(train_loss / train_num)   #将训练的损失放到一个列表里 方便后续画图
    train_accur_all.append(train_accuracy.double().item() / train_num)#训练集的准确率
    test_loss = 0   #同上 测试损失
    test_accuracy = 0.0  #测试准确率
    test_num = 0
    net.eval()   #将模型调整为测试模型
    with torch.no_grad():  #清空历史梯度，进行测试  与训练最大的区别是测试过程中取消了反向传播
        test_bar = tqdm(testdata)
        for data in test_bar:
            img, target = data
            outputs = net(img.to(device))
            loss2 = loss(outputs, target.to(device))
            outputs = torch.argmax(outputs, 1)
            test_loss = test_loss + abs(loss2.item()) * img.size(0)
            accuracy = torch.sum(outputs == target.to(device))
            test_accuracy = test_accuracy + accuracy
            test_num += img.size(0)
            test_acc_save = test_accuracy 
            if i==0:
                test_acc_save = test_accuracy 
                continue
            elif test_accuracy > test_acc_save:
                test_acc_save = test_accuracy 
                style_test = i 
    print("test-Loss：{} , test-accuracy：{}".format(test_loss / test_num, test_accuracy / test_num))
    test_loss_all.append(test_loss / test_num)
    test_accur_all.append(test_accuracy.double().item() / test_num)
    testdev_loss = 0   #同上 测试损失
    testdev_accuracy = 0.0  #测试准确率
    testdev_num = 0
    net.eval()   #将模型调整为测试模型
    with torch.no_grad():  #清空历史梯度，进行测试  与训练最大的区别是测试过程中取消了反向传播
        testdev_bar = tqdm(testdevdata)
        for data in testdev_bar:
            img, target = data
            outputs = net(img.to(device))
            loss3 = loss(outputs, target.to(device))
            outputs = torch.argmax(outputs, 1)
            testdev_loss = testdev_loss + abs(loss3.item()) * img.size(0)
            accuracy = torch.sum(outputs == target.to(device))
            testdev_accuracy = testdev_accuracy + accuracy
            testdev_num += img.size(0)
            testdev_acc_save = testdev_accuracy 
    print("testdev-Loss：{} , testdev-accuracy：{}".format(testdev_loss / testdev_num, testdev_accuracy / testdev_num))
    testdev_loss_all.append(testdev_loss / testdev_num)
    testdev_accur_all.append(testdev_accuracy.double().item() / testdev_num)
 
    val_loss = 0   #同上 测试损失
    val_accuracy = 0.0  #测试准确率
    val_num = 0
    net.eval()   #将模型调整为测试模型
    with torch.no_grad():  #清空历史梯度，进行测试  与训练最大的区别是测试过程中取消了反向传播
        val_bar = tqdm(valdata)
        for data in val_bar:
            img, target = data
 
            outputs = net(img.to(device))
            loss4 = loss(outputs, target.to(device))
            outputs = torch.argmax(outputs, 1)
            val_loss = val_loss + abs(loss4.item()) * img.size(0)
            accuracy = torch.sum(outputs == target.to(device))
            val_accuracy = val_accuracy + accuracy
            val_num += img.size(0)
            if i==0:
                val_acc_save = val_accuracy 
                torch.save(net.state_dict(), "minst_ResNet34.pth")
                continue
            elif val_accuracy > val_acc_save:
                torch.save(net.state_dict(), "minst_ResNet34.pth")
                val_acc_save = val_accuracy 
                style_val = i 
 
    print("val-Loss：{} , val-accuracy：{}".format(val_loss / val_num, val_accuracy / val_num))
    val_loss_all.append(val_loss / val_num)
    val_accur_all.append(val_accuracy.double().item() / val_num)
    torch.save(net.state_dict(), "minst_last.pth")
 
 
#下面的是画图过程，将上述存放的列表  画出来即可
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(epoch), train_loss_all,
         "ro-", label="Train loss")
plt.plot(range(epoch), test_loss_all,
         "bs-", label="test loss")
plt.plot(range(epoch), testdev_loss_all,
         "ys-", label="testdev loss")
plt.plot(range(epoch), val_loss_all,
         "gs-", label="val loss")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.subplot(1, 2, 2)
plt.plot(range(epoch), train_accur_all,
         "ro-", label="Train accur")
plt.plot(range(epoch), test_accur_all,
         "bs-", label="test accur")
plt.plot(range(epoch), testdev_accur_all,
         "ys-", label="testdev accur")
plt.plot(range(epoch), val_accur_all,
         "gs-", label="val accur")
plt.xlabel("epoch")
plt.ylabel("acc")
plt.legend()
plt.show()
 

print("模型已保存")
print(style_val)
print(style_test)