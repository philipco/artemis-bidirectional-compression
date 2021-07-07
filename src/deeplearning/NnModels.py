"""
Created by Philippenko, 26th April 2021.
"""
import torch
from torch import nn
import torch.nn.functional as F

class FullyConnected_Network(nn.Module):
    """
    Fully connected network

    Taken from https://github.com/Xtra-Computing/NIID-Bench/blob/main/model.py
    """

    def __init__(self, input_dim, hidden_dims, output_dim, dropout_p=0.0):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_p = dropout_p

        self.dims = [self.input_dim]
        self.dims.extend(hidden_dims)
        self.dims.append(self.output_dim)

        self.layers = nn.ModuleList([])

        for i in range(len(self.dims) - 1):
            ip_dim = self.dims[i]
            op_dim = self.dims[i + 1]
            self.layers.append(
                nn.Linear(ip_dim, op_dim, bias=True)
            )
        self.__init_net_weights__()

    def __init_net_weights__(self):
        for m in self.layers:
            m.weight.data.normal_(0.0, 0.1)
            m.bias.data.fill_(0.1)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Do not apply ReLU on the final layer
            if i < (len(self.layers) - 1):
                x = F.relu(x)
            if i < (len(self.layers) - 1):  # No dropout on output layer
                x = F.dropout(x, p=self.dropout_p, training=self.training)
        return x

class A9A_Linear(nn.Module):

    def __init__(self):
        input_size = 124
        output_size = 2
        super(A9A_Linear, self).__init__()
        self.l1 = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        x = self.l1(x)
        return x

class A9A_FullyConnected(FullyConnected_Network):

    def __init__(self, dropout_p=0.0):
        super().__init__(input_dim=123, hidden_dims=[32,16,8], output_dim=2)


class Quantum_Linear(nn.Module):

    def __init__(self):
        input_size = 66
        output_size = 2
        super(Quantum_Linear, self).__init__()
        self.l1 = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        x = self.l1(x)
        return x


class Phishing_Linear(nn.Module):

    def __init__(self):
        input_size = 69
        output_size = 2
        super(Phishing_Linear, self).__init__()
        self.l1 = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        x = self.l1(x)
        return x

class Phishing_HiddenLayer(nn.Module):

    def __init__(self):
        input_size = 68
        output_size = 2
        hidden_size = 10
        super(Phishing_HiddenLayer, self).__init__()
        self.hidden_l = nn.Linear(input_size, hidden_size)
        self.predictive_l = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.hidden_l(x))
        x = self.predictive_l(x)
        return x


class MNIST_Linear(nn.Module):
    
    def __init__(self):
        input_size = 784
        output_size = 10
        super(MNIST_Linear, self).__init__()
        self.f1 = nn.Flatten()
        self.l1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.f1(x)
        x = self.l1(x)
        return x


class MNIST_FullyConnected(nn.Module):

    def __init__(self):
        input_size = 784
        output_size = 10
        hidden_size = 128
        super(MNIST_FullyConnected, self).__init__()
        self.f1 = nn.Flatten()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.l3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.f1(x)
        x = self.l1(x)
        x = self.tanh(x)
        x = self.l3(x)
        return x


class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class FashionMNIST_CNN(nn.Module):
    """From https://www.kaggle.com/pankajj/fashion-mnist-with-pytorch-93-accuracy"""

    def __init__(self):
        super(FashionMNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 15, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(15, 30, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(1080, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        # conv1(kernel=3, filters=15) 28x28x1 -> 26x26x15
        x = F.relu(self.conv1(x))

        # conv2(kernel=3, filters=20) 26x26x15 -> 13x13x30
        # max_pool(kernel=2) 13x13x30 -> 6x6x30
        x = F.relu(F.max_pool2d(self.conv2(x), 2, stride=2))

        # flatten 6x6x30 = 1080
        x = x.view(-1, 1080)

        # 1080 -> 100
        x = F.relu(self.fc1(x))

        # 100 -> 10
        x = self.fc2(x)

        # transform to logits
        return x


class FEMNIST_CNN(nn.Module):
    """Model for FEMNIST

    This class is taken from the code of paper Federated Learning on Non-IID Data Silos: An Experimental Study.
    https://github.com/Xtra-Computing/NIID-Bench
    """

    def __init__(self):
        super(FEMNIST_CNN, self).__init__()
        input_dim = (16 * 4 * 4)
        hidden_dims = [120, 84]
        output_dim = 10
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet(nn.Module):
    """From https://github.com/kuangliu/pytorch-cifar/blob/master/models/lenet.py."""
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.tanh(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = self.tanh(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.tanh(self.fc1(out))
        out = self.tanh(self.fc2(out))
        out = self.fc3(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    """From https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py"""
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])
