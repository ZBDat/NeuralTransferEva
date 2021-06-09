import os
from matplotlib import pyplot as plt
import torch
import torchvision
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from LoadData import load_data

INPUT_DIR = os.getcwd()
LABEL_FILE = 'Data\\ROISignals_FunRawRWSDCF\\Patient_Information_2019_09_29.xlsx'
DATA_DIR = os.path.abspath('.\\Data\\ROISignals_FunRawRWSDCF')
SAVE_PATH = '.\\trained_model.pth'


class Conv2dNet(nn.Module):

    def __init__(self):
        super(Conv2dNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.pool = nn.MaxPool2d(5, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.fc1 = nn.Linear(480, 200)
        self.fc2 = nn.Linear(200, 75)
        self.fc3 = nn.Linear(75, 15)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.25)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def train(train_loader):
    net = Conv2dNet().float()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.01)

    for epoch in range(50):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data['signal']
            labels = data['label']

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 5 == 4:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 5))
                running_loss = 0.0

    print('Finished Training')
    torch.save(net.state_dict(), SAVE_PATH)


def test(test_loader):
    net = Conv2dNet()
    net.load_state_dict(torch.load(SAVE_PATH))

    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            signals = data['signal']
            labels = data['label']

            outputs = net(signals)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


if __name__ == "__main__":
    classes = ('no', 'yes')
    train_loader, test_loader = load_data(INPUT_DIR, LABEL_FILE, DATA_DIR)
    train(train_loader)
    test(test_loader)
