import torch
import random
import numpy as np
from collections import deque
from snake_model import Linear_QNet, QTrainer
from snake_game import runAI, direct, pt
# from SnakeModel import QTrainer
from snake_plot import plot

maxmem = 100_000
batchsz = 1000
learn_rate = 0.001

class Agent:
    def __init__(self):
        self.num_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate < 1
        self.mem = deque(maxlen=maxmem) # popleft()

        #  model trainer
        # size of state, x, output
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr= learn_rate, gamma = self.gamma)


    def state(self,game):
        head = game.body[0]
        point_l = pt(head.x - 20, head.y)
        point_r = pt(head.x + 20, head.y)
        point_u = pt(head.x, head.y - 20)
        point_d = pt(head.x, head.y + 20)

        dir_l = game.direct == direct.left
        dir_r = game.direct == direct.right
        dir_u = game.direct == direct.up
        dir_d = game.direct == direct.down

        state = [
            # Danger straight
            (dir_r and game.died(point_r)) or
            (dir_l and game.died(point_l)) or
            (dir_u and game.died(point_u)) or
            (dir_d and game.died(point_d)),

            # Danger right
            (dir_u and game.died(point_r)) or
            (dir_d and game.died(point_l)) or
            (dir_l and game.died(point_u)) or
            (dir_r and game.died(point_d)),

            # Danger left
            (dir_d and game.died(point_r)) or
            (dir_u and game.died(point_l)) or
            (dir_r and game.died(point_u)) or
            (dir_l and game.died(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.obj.x < game.front.x,  # food left
            game.obj.x > game.front.x,  # food right
            game.obj.y < game.front.y,  # food up
            game.obj.y > game.front.y  # food down
            ]

        # convert t/f to 0/1
        return np.array(state, dtype=int)


    def memory(self, state, action, reward, nextstate, done):
        # one tuple
        self.mem.append((state, action, reward, nextstate, done)) #popleft if maxmem reached

    def train_longmem(self):
        if len(self.mem) > batchsz:
            # list of tuples
            minisample = random.sample(self.mem, batchsz)
        else:
            minisample = self.mem

        states, actions, rewards, nextstates, dones = zip(*minisample)
        # could use a for loop instead
        self.trainer.train_st(states, actions, rewards, nextstates, dones)



    def train_shortmem(self, state, action, reward, nextstate, done):
        # train for one game step
        self.trainer.train_st(state, action, reward, nextstate, done)

    def get_act(self, state):
        # random moves
        # deep learning: tradeoff exploration v exploitation
        self.epsilon = 80 - self.num_games
        # more games, smaller epsilon
        fin_move = [0,0,0]
        # smaller epsilon, less frequent
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,2)
            fin_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            # convert max/min to integer
            move = torch.argmax(prediction).item()
            fin_move[move]=1

        return fin_move



def train():
    plot_scores = []
    plot_meanscores = []
    tot_score = 0
    record = 0
    ag = Agent()
    game = runAI()
    while True:
        # find old state
        old_state = ag.state(game)

        # find next move
        fin_move = ag.get_act(old_state)

        # do move and get new state
        reward, done, score = game.play(fin_move)
        new_state = ag.state(game)

        # train short mem
        ag.train_shortmem(old_state, fin_move, reward, new_state, done)

        # remember

        ag.memory(old_state, fin_move, reward, new_state, done)

        if done:
            # train longmem (experience replay) (trains on all prev moves/games)
            game.reset()
            ag.num_games +=1
            ag.train_longmem()

            if score > record:
                record = score
                ag.model.save()

            print('Game', ag.num_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            tot_score += score
            mean_score = tot_score / ag.num_games
            plot_meanscores.append(mean_score)
            plot(plot_scores, plot_meanscores)


if __name__ == '__main__':
    train()