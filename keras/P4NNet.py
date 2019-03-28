import sys
sys.path.append('..')
from utils import *

import os
import argparse
from keras.models import *
from keras.layers import *
from keras.optimizers import *

class P4NNet():
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args


        dim = (self.board_x-4)*(self.board_y-4)*256 + self.board_x*self.board_y
        inp = Input(shape=((dim,)))
        
        x = Activation('relu')(Dense(256)(inp)) 
        
        self.pi = Dense(self.action_size, activation='softmax', name='pi')(x)   # batch_size x self.action_size
        self.v = Dense(1, activation='tanh', name='v')(x)                    # batch_size x 1

        fpath = os.path.join('./temp/','best.pth.tar')
        if not os.path.exists(fpath):
            raise("No model in path {}".format(fpath))
            
        
        
        self.model = Model(inputs=inp, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(args.lr))
        self.model.load_weights(fpath)
