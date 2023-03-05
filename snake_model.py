import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module):

    def __init__(self, inputsz, hiddensz, outputsz):
        super().__init__()
        self.linear1 = nn.Linear(inputsz, hiddensz)
        self.linear2 = nn.Linear(hiddensz, outputsz)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, filename='model.pth'):
        model_folderpath = './model'
        if not os.path.exists(model_folderpath):
            os.makedirs(model_folderpath)

        filename = os.path.join(model_folderpath, filename)
        torch.save(self.state_dict(), filename)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # loss function, mean squared error
        self.criterion = nn.MSELoss()

    def train_st(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # only have 1 dimension/number, but want (numbatch, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
            # tuple with 1 value

        # 1: predicted Q VAL w/ current state
        pred = self.model(state)

        target = pred.clone()
        for windex in range(len(done)):
            Qnew = reward[windex]
            if not done[windex]:
                Qnew = reward[windex] + self.gamma + torch.max(self.model(next_state[windex]))

            target[windex][torch.argmax(action[windex]).item()] = Qnew

        # 2: Q_new = r + y * max(predicted Q val)- only do if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
