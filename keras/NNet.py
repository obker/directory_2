import argparse
import os
import shutil
import time
import random
import numpy as np
import math
import sys
import tensorflow as tf
sys.path.append('..')
from utils import *
from NeuralNet import NeuralNet

import argparse
from .P4NNet import P4NNet as onnet
from .P4NNet5x5 import P4NNet as onnet5x5

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': True,
    'num_channels': 64,
})

args5x5 = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': True,
    'num_channels': 64,
})


class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.graph = tf.get_default_graph()
        self.nnet = onnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.nnet5x5 = onnet5x5(game, args5x5)
                            

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        
        features = input_boards.reshape(input_boards.shape[0],-1)
        for i in range(self.board_x-4):
            for j in range(self.board_y-4):
                b = input_boards[:,i:i+5,j:j+5]
                with self.graph.as_default():
                    self.nnet5x5.newmodel._make_predict_function()
                    f = self.nnet5x5.newmodel.predict(b)
                features = np.concatenate((features, f), axis=1)
                
        self.nnet.model.fit(x = features, y = [target_pis, target_vs], batch_size = args.batch_size, epochs = args.epochs)

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        # start = time.time()

        # preparing input
        features = board.reshape(-1)
        for i in range(self.board_x-4):
            for j in range(self.board_y-4):
                b = board[i:i+5,j:j+5]
                b = b[np.newaxis,:,:]
                with self.graph.as_default():
                    self.nnet5x5.newmodel._make_predict_function()
                    f = self.nnet5x5.newmodel.predict(b)[0]
                features = np.concatenate((features, f), axis=0)
        features = features[np.newaxis,:]   
        with self.graph.as_default():
            # run
            self.nnet.model._make_predict_function()
            pi, v = self.nnet.model.predict(features)

        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi[0], v[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))
        self.nnet.model.load_weights(filepath)
