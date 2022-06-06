import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random

class LSTM_RUL(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout, bid, device):
        super(LSTM_RUL, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bid = bid
        self.dropout = dropout
        self.device = device
        # encoder definition
        self.encoder = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=self.bid)
        # regressor
        self.regressor= nn.Sequential(
            nn.Linear(self.hidden_dim+self.hidden_dim*self.bid, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, 1))

    def forward(self, src):
        encoder_outputs, (hidden, cell) = self.encoder(src)
        # select the last hidden state as a feature
        features = encoder_outputs[:, -1:].squeeze()
        predictions = self.regressor(features)
        return predictions.squeeze(), features


class Generator(nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.encoder_1 = nn.Sequential(
            nn.Conv1d(self.input_dim, self.input_dim, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.LeakyReLU(),
            # nn.Tanh(),
            nn.Conv1d(self.input_dim, self.input_dim, kernel_size=3, stride=2, padding=1, dilation=4),
            nn.LeakyReLU(),
            # nn.Tanh(),
            nn.MaxPool1d(kernel_size=5)
        )
        self.encoder_2 = nn.Sequential(
            nn.Conv1d(self.input_dim, self.input_dim, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.LeakyReLU(),
            # nn.Tanh(),
            nn.Conv1d(self.input_dim, self.input_dim, kernel_size=5, stride=2, padding=1, dilation=2),
            nn.LeakyReLU(),
            # nn.Tanh(),
            nn.MaxPool1d(kernel_size=5)
        )
        self.encoder_3 = nn.Sequential(
            nn.Conv1d(self.input_dim, self.input_dim, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.LeakyReLU(),
            # nn.Tanh(),
            nn.Conv1d(self.input_dim, self.input_dim, kernel_size=7, stride=2, padding=1, dilation=1),
            nn.LeakyReLU(),
            # nn.Tanh(),
            nn.MaxPool1d(kernel_size=6)
        )
        self.flat = nn.Flatten()
        self.adapter = nn.Linear(42, 64)  # output = teacher network feature output

    def forward(self, src):
        fea_1 = self.encoder_1(src)
        fea_2 = self.encoder_2(src)
        fea_3 = self.encoder_3(src)
        features = self.flat(torch.cat((fea_1,fea_2,fea_3),dim=2))
        hint = self.adapter(features)
        return features, hint


class CNN_RUL_student_stack(nn.Module):
    def __init__(self, input_dim, hidden_dim,dropout):
        super(CNN_RUL_student_stack, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.dropout=dropout
        self.generator = Generator(input_dim=input_dim)
        self.regressor= nn.Sequential(
            nn.Linear(64, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, src):
        features,hint = self.generator(src)
        predictions = self.regressor(hint)
        return predictions.squeeze(), features, hint


class MyDataset(Dataset):
    def __init__(self, x, y):
        super(MyDataset, self).__init__()
        self.len = x.shape[0]
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.x_data = torch.as_tensor(x, device=device, dtype=torch.float)
        self.y_data = torch.as_tensor(y, device=device, dtype=torch.float)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], index

    def __len__(self):
        return self.len


def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))


def Score(yhat,y):
    score =0
    for i in range(len(yhat)):
        if y[i] <= yhat[i]:
            score = score + torch.exp(-(y[i]-yhat[i])/10.0)-1
        else:
            score = score + torch.exp((y[i]-yhat[i])/13.0)-1
    return score.item()


def centered_average(nums):
    return sum(nums) / len(nums)


def weights_init(m):
    for child in m.children():
        if isinstance(child,nn.Linear) or isinstance(child,nn.Conv1d):
            torch.nn.init.xavier_uniform_(child.weight)


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon))
            bias = self.bias_mu + self.bias_sigma.mul(Variable(self.bias_epsilon))
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x


class Qnet(nn.Module):
    def __init__(self, input_dim):
        super(Qnet, self).__init__()

        self.input_dim = input_dim
        self.linear1 = nn.Linear(self.input_dim, 1024) # Input size is state size 64 + 1 + 1

        self.noisy_value1 = NoisyLinear(1024, 1024)
        self.noisy_value2 = NoisyLinear(1024, 1)

        self.noisy_advantage1 = NoisyLinear(1024, 1024)
        self.noisy_advantage2 = NoisyLinear(1024, 2)

    def forward(self, x):
        x1 = F.relu(self.linear1(x))
        value1 = F.relu(self.noisy_value1(x1))
        value = self.noisy_value2(value1)

        advantage1 = F.relu(self.noisy_advantage1(x1))
        advantage = self.noisy_advantage2(advantage1)
        return value + advantage - advantage.mean() # dim = [batch_size, 2]

    def sample_action(self, obs):
        out = self.forward(obs)
        # if Q value of action 1 == Q_value of action_2
        if out[0, 0] == out[0, 1]:
            return np.array([random.randrange(2)]), out, 2
        else:
            # return shape ndarray (batch_size,)
            return (torch.argmax(out, dim=1)).cpu().detach().numpy(), out, 0

    def reset_noise(self):
        self.noisy_value1.reset_noise()
        self.noisy_value2.reset_noise()
        self.noisy_advantage1.reset_noise()
        self.noisy_advantage2.reset_noise()


class Dueling_DQN(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(Dueling_DQN, self).__init__()
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=7, stride=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.fc1_adv = nn.Linear(in_features=6 * 6 * 64, out_features=512)
        self.fc1_val = nn.Linear(in_features=6 * 6 * 64, out_features=512)

        self.fc2_adv = nn.Linear(in_features=512, out_features=num_actions)
        self.fc2_val = nn.Linear(in_features=512, out_features=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        adv = self.relu(self.fc1_adv(x))
        val = self.relu(self.fc1_val(x))

        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(x.size(0), self.num_actions)

        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)
        return x

    def sample_action(self, obs):
        out = self.forward(obs)
        return (torch.argmax(out, dim=1)).detach().cpu().numpy(), out, 0