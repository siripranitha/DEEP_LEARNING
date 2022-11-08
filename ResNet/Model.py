import os
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from NetWork import ResNet
from ImageUtils import parse_record

""" This script defines the training, validation and testing process.
"""

class Cifar(nn.Module):
    def __init__(self, config):
        super(Cifar, self).__init__()
        self.config = config
        self.network = ResNet(
            self.config.resnet_version,
            self.config.resnet_size,
            self.config.num_classes,
            self.config.first_num_filters,
        )
        self.lr = 0.01
        self.learning_rate = 0.01
        ### YOUR CODE HERE
        # define cross entropy loss and optimizer
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.network.parameters(),  momentum=0.9,
                                     weight_decay=self.config.weight_decay,
                                     lr=self.learning_rate# should i take it in input?
                                     )
        #if torch.cuda.is_available():
        torch.cuda.set_device('cuda:0')
        self.network = self.network.cuda()
        self.cross_entropy_loss = self.cross_entropy_loss.cuda()


        ### YOUR CODE HERE

    def train(self, x_train, y_train, max_epoch):
        self.network.train()
        # Determine how many batches in an epoch
        num_samples = x_train.shape[0]
        num_batches = num_samples // self.config.batch_size

        print('### Training... ###')
        for epoch in range(1, max_epoch+1):
            start_time = time.time()
            # Shuffle
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]

            ### YOUR CODE HERE
            # Set the learning rate for this epoch
            # Usage example: divide the initial learning rate by 10 after several epochs
            if (epoch % 20 == 0):
                self.learning_rate = self.learning_rate * (0.1 ** (epoch // 20))
            
            ### YOUR CODE HERE
            
            for i in range(num_batches):
                ### YOUR CODE HERE
                # Construct the current batch.
                # Don't forget to use "parse_record" to perform data preprocessing.
                # Don't forget L2 weight decay
                input_raw = curr_x_train[i * self.config.batch_size:(i + 1) * self.config.batch_size]
                input = np.apply_along_axis(lambda x: parse_record(x, True), 1, input_raw)
                self.optimizer.zero_grad()

                #if torch.cuda.is_available():
                    #print('cuda is available.')
                output = self.network(torch.tensor(input).to('cuda'))
                target = curr_y_train[i * self.config.batch_size:(i + 1) * self.config.batch_size]
                loss = self.cross_entropy_loss(output, torch.tensor(target).long().to('cuda'))
                #else:
                #    output = self.network(torch.tensor(input))
                #    target = curr_y_train[i * self.config.batch_size:(i + 1) * self.config.batch_size]
                #    loss = self.cross_entropy_loss(output, torch.tensor(target).long())


                # accuracy, acc5 = self.accuracy(output, target, topk=(1, 5))

                ### YOUR CODE HERE

                loss.backward()
                self.optimizer.step()

                print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss), end='\r', flush=False)
            
            duration = time.time() - start_time
            print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, loss, duration))

            if epoch % self.config.save_interval == 0:
                self.save(epoch)


    def test_or_validate(self, x, y, checkpoint_num_list):
        self.network.eval()
        print('### Test or Validation ###')
        for checkpoint_num in checkpoint_num_list:
            checkpointfile = os.path.join(self.config.modeldir, 'model-%d.ckpt'%(checkpoint_num))
            self.load(checkpointfile)

            preds = []
            for i in tqdm(range(x.shape[0])):
                ### YOUR CODE HERE
                current_input=parse_record(x[i], False)
                output= self.network(torch.tensor(current_input).float().view(1, 3, 32, 32).to('cuda'))
                _, preds  = torch.max(output, dim=1)
                ### END CODE HERE

            y = torch.tensor(y).to('cuda')
            preds = torch.tensor(preds).to('cuda')
            print('Test accuracy: {:.4f}'.format(torch.sum(preds==y)/y.shape[0]))
    
    def save(self, epoch):
        checkpoint_path = os.path.join(self.config.modeldir, 'model-%d.ckpt'%(epoch))
        os.makedirs(self.config.modeldir, exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Checkpoint has been created.")
    
    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))