### YOUR CODE HERE
# import tensorflow as tf
# import torch
import os
from time import time
import numpy as np
from Network import MyNetwork
from ImageUtils import parse_record

import torch
import tqdm


"""This script defines the training, validation and testing process.
"""

class MyModel(object):

    def __init__(self, configs):
        self.configs = configs
        self.network = MyNetwork(configs)

    def model_setup(self):
        # todo: setting up the checkpoint model occurs here. model setup
        self.loss = torch.nn.CrossEntropyLoss()
        self.learning_rate = self.configs.learning_rate
        #TODO: INCLUDE SGD IN REPORT!
        self.optimizer = torch.optim.SGD(self.network.parameters(),
                                         lr = self.learning_rate,
                                         momentum=self.configs.momentum,
                                         weight_decay=self.configs.weight_decay
                                         )
        # learning rate scheduler. TODO: TRY OUT DIFFERNT TYPES
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=200)

    def train(self, x_train, y_train, configs, x_valid=None, y_valid=None):
        # todo: using the validation.
        self.network.train() # todo: test how the call thing works in network

        num_samples = x_train.shape[0]
        num_batches = num_samples // self.configs.batch_size

        print('### Training... ###')
        for _epoch in range(self.configs.max_epoch):
            st = time()
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train,curr_y_train = x_train[shuffle_index],y_train[shuffle_index]
            for i in range(num_batches):



                current_input_x = curr_x_train[i * self.configs.batch_size:(i + 1) * self.configs.batch_size]
                current_input_x = np.apply_along_axis(lambda x: parse_record(x, True), 1, current_input_x)
                current_input_x = torch.stack(current_input_x).float()#.cuda()

                current_input_y = curr_y_train[i * self.configs.batch_size:(i + 1) * self.configs.batch_size]

                self.optimizer.zero_grad()
                output = self.network(current_input_x)#.to('cuda')) #todo: network, cuda conversion, torch stack

                current_loss = self.loss(output, torch.tensor(current_input_y).long())#.to('cuda'))
                current_loss.backward()
                self.optimizer.step()

                print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, current_loss), end='\r', flush=False)
            self.scheduler.step()
            print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(_epoch+1, current_loss, time()-st))

            if (_epoch+1)%self.configs.save_interval==0 or (_epoch>178 and current_loss<0.001):
                self.save(_epoch+1)

    def save(self,epoch):
        save_path = os.path.join(self.configs.save_dir,f'model-{str(epoch)}.ckpt')
        torch.save(dict(
            epoch=epoch,
            model_state=self.network.state_dict(),#todo:
            optimizer_state = self.optimizer.state_dict(),
        ),save_path)

    def load(self,path):
        checkpoint = torch.load(path,map_location='cpu')
        self.network.load_state_dict(checkpoint,strict=True) #todo:
        print(f'produced model params from {path}')

    def evaluate(self, x, y):
        pass

    def predict_prob(self, x):
        pass


### END CODE HERE