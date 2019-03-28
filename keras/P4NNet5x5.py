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
        self.board_x, self.board_y = 5,5
        self.action_size = 26
        self.args = args

        # Neural Net
        self.input_boards = Input(shape=(self.board_x, self.board_y))    # s: batch_size x board_x x board_y

        x_image = Reshape((self.board_x, self.board_y, 1))(self.input_boards)                # batch_size  x board_x x board_y x 1
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 2, padding='same', trainable= False)(x_image)))         # batch_size  x board_x x board_y x num_channels
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same', trainable= False)(h_conv1)))         # batch_size  x board_x x board_y x num_channels
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same', trainable= False)(h_conv2)))         # batch_size  x board_x x board_y x num_channels
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='valid',trainable= False)(h_conv2)))        # batch_size  x (board_x-2) x (board_y-2) x num_channels
        h_conv4_flat = Flatten()(h_conv4)       
        s_fc1 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(256, trainable = False)(h_conv4_flat))))  # batch_size x 1024
        s_fc2 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(256, trainable = False)(s_fc1))))          # batch_size x 1024
        self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)   # batch_size x self.action_size
        self.v = Dense(1, activation='tanh', name='v')(s_fc2)                    # batch_size x 1

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(args.lr))

        fpath = os.path.join('./p4/keras/','best5x5.pth.tar')
        if not os.path.exists(fpath):
            raise("No model in path {}".format(fpath))
            
        self.model.load_weights(fpath)
        self.model.layers.pop()
        
        self.newmodel = Model(inputs = self.input_boards, output = s_fc2)
        self.newmodel.compile(optimizer = Adam(args.lr), loss = 'mse')
        self.newmodel.set_weights(self.model.get_weights())
