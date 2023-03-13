import torch
import torchvision
import numpy as np
from cnn import SparseCNN

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(
), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
trainset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
testset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=64,
    shuffle=True,
    num_workers=2
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=64,
    shuffle=False,
    num_workers=2
)

criterion = torch.nn.CrossEntropyLoss()  # 定义交叉熵损失函数对象


def train_model(model: SparseCNN, total_worker_num:int, curr_worker_num: int):
    epochs = 1  # 定义训练轮数
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 定义随机

    for epoch in range(epochs):  # 遍历每一轮
        running_loss = 0.0  # 初始化累计损失为0
        for i, data in enumerate(trainloader):  # 遍历每一个批次
            if i % 200 == 0 and i != 0:  # 每200个批次打印一次平均损失值
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i, running_loss / 200))
                running_loss = 0.0
            if i % total_worker_num != curr_worker_num:
                continue # 每个节点训练不同的数据
            inputs, labels = data  # 获取输入和标签
            optimizer.zero_grad()  # 清零梯度缓存
            outputs = model(inputs)  # 计算模型输出
            loss = criterion(outputs, labels)  # 计算损失函数值
            loss.backward()  # 反向传播梯度
            optimizer.step()  # 更新模型参数

            running_loss += loss.item()  # 累加损失值
            


def test_model(model: SparseCNN):
    correct = 0  # 初始化正确预测数为0
    total = 0  # 初始化总样本数为0
    with torch.no_grad():  # 不需要计算梯度
        for data in testloader:  # 遍历每一个批次
            images, labels = data  # 获取输入和标签
            outputs = model(images)  # 计算模型输出
            _, predicted = torch.max(outputs.data, 1)  # 获取最大概率的类别作为预测结果
            total += labels.size(0)  # 累加总样本数
            correct += (predicted == labels).sum().item()  # 累加正确预测数
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total
    ))
