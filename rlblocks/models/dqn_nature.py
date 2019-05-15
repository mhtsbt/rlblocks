import torch
import torch.nn as nn
import torch.nn.functional as f


class DQNNatureModel(nn.Module):

    def _feature_size(self):
        # feed some sample data into the network, in order to determine the receptive field
        sample = torch.zeros(size=(1, *self.input_shape))
        return self.conv3(self.conv2(self.conv1(sample))).view(1, -1).size(1)

    def __init__(self, actions, input_shape):
        super(DQNNatureModel, self).__init__()
        self.input_shape = input_shape

        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc4 = nn.Linear(self._feature_size(), 512) # TODO: calculate this
        self.fc5 = nn.Linear(512, actions)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.relu(self.conv2(x))
        x = f.relu(self.conv3(x))
        x = f.relu(self.fc4(x.view(x.size(0), -1)))

        return self.fc5(x)

