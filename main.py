import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
import torch.nn.functional as F

from sklearn.model_selection import train_test_split

# 定义自定义的神经网络模型
class MyNNModel(nn.Module):
    def __init__(self):
        super(MyNNModel, self).__init__()
        
        # 第一层卷积层，输入通道数为1，输出通道数为16，卷积核大小为3，填充为1
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        # 第二层卷积层，输入通道数为16，输出通道数为32，卷积核大小为3，填充为1
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)  # 最大池化层，核大小为2，步幅为2
        
        # 全连接层，输入特征数为32*7*7，输出特征数为500
        self.fc1 = nn.Linear(32 * 7 * 7, 500)
        self.fc2 = nn.Linear(500, 10)  # 输出特征数为10，对应分类类别数
        
        self.dropout = nn.Dropout(0.25)  # Dropout层，用于防止过拟合
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 第一层卷积，ReLU激活，池化
        x = self.pool(F.relu(self.conv2(x)))  # 第二层卷积，ReLU激活，池化
        
        x = x.view(-1, 32 * 7 * 7)  # 将特征展平
        x = self.dropout(x)  # Dropout层
        x = F.relu(self.fc1(x))  # 全连接层，ReLU激活
        x = self.dropout(x)  # Dropout层
        x = F.log_softmax(self.fc2(x), dim=1)  # 输出层，使用LogSoftmax激活函数
        
        return x

# 数据预处理函数，用于将输入数据转换为模型可接受的格式
def data_transform(X, y_hat=None):
    
    if y_hat is not None:
        # 处理训练数据
        X = X.values.reshape(-1, 1, 28, 28)  # 重新调整数据形状为通道 x 高度 x 宽度
        X = torch.from_numpy(X)  # 转换为PyTorch张量
        X = torch.true_divide(X, 255)  # 归一化到[0, 1]
        X = torch.true_divide((X - 0.5), 0.5)  # 归一化到[-1, 1]
        
        y_hat = torch.from_numpy(y_hat.values)  # 将标签转换为PyTorch张量
        y_hat.type(torch.LongTensor)  # 转换标签类型为长整型
        
        return X, y_hat
    
    else:
        # 处理测试数据
        X = X.values.reshape(-1, 1, 28, 28)  # 重新调整数据形状为通道 x 高度 x 宽度
        X = torch.from_numpy(X)  # 转换为PyTorch张量
        X = torch.true_divide(X, 255)  # 归一化到[0, 1]
        X = torch.true_divide((X - 0.5), 0.5)  # 归一化到[-1, 1]
        
        return X

# 读取训练和测试数据集
data_train = pd.read_csv("data/train.csv")
data_test = pd.read_csv("data/test.csv")

X_raw = data_train.drop(columns=["label"])  # 特征数据
y_raw = data_train["label"]  # 标签数据

# 划分训练集和验证集
X_train, X_valid, y_train, y_valid = train_test_split(X_raw, y_raw, test_size=0.20, random_state=42)

# 数据预处理
X_train_transform, y_train_transform = data_transform(X=X_train, y_hat=y_train)
X_valid_transform, y_valid_transform = data_transform(X=X_valid, y_hat=y_valid)
X_test_transform = data_transform(data_test)

batch_size = 32

# 创建训练集和验证集的数据加载器
data_train = torch.utils.data.TensorDataset(X_train_transform, y_train_transform)
data_valid = torch.utils.data.TensorDataset(X_valid_transform, y_valid_transform)
loader_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size)
loader_valid = torch.utils.data.DataLoader(data_valid, batch_size=batch_size)

# 创建测试集的数据加载器
data_test = torch.utils.data.TensorDataset(X_test_transform)
loader_test = torch.utils.data.DataLoader(data_test)

# 创建模型实例
model = MyNNModel()

if torch.cuda.is_available():
    model.cuda()

print(model)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

# 训练模型
epochs = 1
valid_loss_min = np.Inf
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

train_losses = []
valid_losses = []

for epoch in range(1, epochs+1):
    # 跟踪训练和验证损失
    loss_train = 0.0
    loss_valid = 0.0
    
    # 设置为训练模式，启用Dropout
    model.train()
    for pic, target in loader_train:
        if torch.cuda.is_available():
            pic, target = pic.cuda(), target.cuda()
        optimizer.zero_grad()  # 清除梯度
        output = model(pic)  # 前向传播
        loss = criterion(output, target)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重
        loss_train += loss.item() * pic.size(0)
    
    # 设置为评估模式，关闭Dropout，禁止计算梯度
    model.eval()
    with torch.no_grad():
        for pic, target in loader_valid:
            if torch.cuda.is_available():
                pic, target = pic.cuda(), target.cuda()
            output = model(pic)  # 前向传播
            loss = criterion(output, target)  # 计算损失
            loss_valid += loss.item() * pic.size(0)
            
            _, pred = torch.max(output, 1)
            correct = np.squeeze(pred.eq(target.data.view_as(pred)))
            
            for i in range(len(target)):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1
            
    loss_train = loss_train / len(loader_train.sampler)
    loss_valid = loss_valid / len(loader_valid.sampler)
    
    train_losses.append(loss_train)
    valid_losses.append(loss_valid)
    
    # 如果验证损失减小，保存模型
    if loss_valid <= valid_loss_min:
        print(f"Validation loss decreased ({valid_loss_min:.6f} --> {loss_valid:.6f})."
              "Saving model ...")
        torch.save(model, "model/CNN.pt")
        valid_loss_min = loss_valid
        
# 打印每个类别的测试准确率
for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            str(i), 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

# 打印整体的测试准确率
print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))

# 加载保存的模型
model = torch.load("model-test/CNN.pt", map_location=torch.device('cpu'))

if torch.cuda.is_available():
    model.cuda()

print(model)

labels = []
model.eval()
with torch.no_grad():
    for pic in loader_test:
        pic = pic[0]
        
        if torch.cuda.is_available():
            pic = pic.cuda()
        
        output = model(pic)
        labels.append(torch.argmax(torch.exp(output).data).item())

# 生成提交文件
submission_dict = {'ImageId': list(range(1, len(loader_test.sampler) + 1)), 'Label': labels}
df_submission = pd.DataFrame.from_dict(submission_dict)
df_submission.to_csv('output/submission.csv', index=False)
