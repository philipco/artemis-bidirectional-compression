"""
Created by Philippenko, 26th April 2021.

All neural network use for DL.
"""
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.loss import _WeightedLoss
from torchvision.models.squeezenet import Fire


class LogisticLoss(_WeightedLoss):
    """This class is used to obtain almost exact results as without neural networks.

    To use it, it requires to remove the sigmoid from the neural network, to replace BCELoss by this one, and to set
    targets values to +/-1. Futhermore, to compute the accuracy needs to use torch.sign(output) instead of round().
    Concerning the dataset, targets should be of the form: torch.cat([y for y in Y_train])
    """
    def __init__(self):
        super(LogisticLoss, self).__init__()

    def forward(self, input, target):
        n_samples = len(input)
        return -torch.sum(torch.log(torch.sigmoid(target * input.flatten()))) / n_samples


class LogisticReg(nn.Module):

    def __init__(self, input_size):
        self.output_size = 1
        super(LogisticReg, self).__init__()
        self.l1 = nn.Linear(input_size, self.output_size, bias=False)

    def forward(self, x):
        x = self.l1(x)
        return torch.sigmoid(x)


class MNIST_Linear(nn.Module):
    
    def __init__(self, input_size):
        input_size = 784
        self.output_size = 10
        super(MNIST_Linear, self).__init__()
        self.f1 = nn.Flatten()
        self.l1 = nn.Linear(input_size, self.output_size)

    def forward(self, x):
        x = self.f1(x)
        x = self.l1(x)
        return x


class MNIST_FullyConnected(nn.Module):

    def __init__(self, input_size):
        input_size = 784
        self.output_size = 10
        hidden_size = 128
        super(MNIST_FullyConnected, self).__init__()
        self.f1 = nn.Flatten()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.l3 = nn.Linear(hidden_size, self.output_size)

    def forward(self, x):
        x = self.f1(x)
        x = self.l1(x)
        x = self.tanh(x)
        x = self.l3(x)
        return x


class MNIST_CNN(nn.Module):
    def __init__(self, input_size):
        super(MNIST_CNN, self).__init__()
        self.output_size = 10
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, self.output_size)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class FashionSimpleNet(nn.Module):
    """ From https://github.com/kefth/fashion-mnist/blob/master/model.py"""

    def __init__(self, input_size):
        super().__init__()
        self.output_size = 10
        self.features = nn.Sequential(
            nn.Conv2d(1,32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.output_size)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 64 * 7 * 7)
        x = self.classifier(x)
        return x


class FashionMNIST_CNN(nn.Module):
    """From https://www.kaggle.com/pankajj/fashion-mnist-with-pytorch-93-accuracy"""

    def __init__(self, input_size):
        super(FashionMNIST_CNN, self).__init__()
        self.output_size = 10
        self.conv1 = nn.Conv2d(1, 15, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(15, 30, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(1080, 100)
        self.fc2 = nn.Linear(100, self.output_size)

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

    def __init__(self, input_size):
        super(FEMNIST_CNN, self).__init__()
        input_dim = (16 * 4 * 4)
        hidden_dims = [120, 84]
        self.output_size = 10
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], self.output_size)

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
    def __init__(self, input_size):
        super(LeNet, self).__init__()
        self.output_size = 10
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, self.output_size)
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


class SqueezeNet(nn.Module):

    def __init__(self, input_size, version='1_0', num_classes=10):
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
        self.output_size = 10
        if version == '1_0':
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        elif version == '1_1':
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        else:
            # FIXME: Is this needed? SqueezeNet should only be called from the
            # FIXME: squeezenet1_x() functions
            # FIXME: This checking is not done for the other models
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1_0 or 1_1 expected".format(version=version))

        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)
