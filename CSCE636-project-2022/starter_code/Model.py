### YOUR CODE HERE
# import tensorflow as tf
# import torch
import os
import pickle
from time import time
import numpy as np
from Network import MyNetwork
from ImageUtils import parse_record

import torch
import torch.nn.functional as F
from tqdm import tqdm


"""This script defines the training, validation and testing process.
"""

class MyModel(object):

    def __init__(self, configs):
        self.configs = configs
        self.network = MyNetwork(resnet_size=configs.resnet_size,
                 num_classes=10,
                 width=configs.width)
        self.model_setup()

    def model_setup(self):

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
        if torch.cuda.is_available():
            self.network = self.network.cuda()
            self.loss = self.loss.cuda()

    def train(self, x_train, y_train, configs, x_valid=None, y_valid=None):
        # todo: using the validation.
        self.network.train() # todo: test how the call thing works in network

        num_samples = x_train.shape[0]
        num_batches = num_samples // self.configs.batch_size

        print('### Training... ###')
        current_loss=None
        results = []
        for _epoch in range(self.configs.max_epoch):
            st = time()
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train,curr_y_train = x_train[shuffle_index],y_train[shuffle_index]


            for i in range(num_batches):
                current_input_x = curr_x_train[i * self.configs.batch_size:(i + 1) * self.configs.batch_size]
                current_input_x = [parse_record(x, True) for x in current_input_x]
                current_input_x = torch.stack(current_input_x).float()#.cuda()
                current_input_y = torch.tensor(curr_y_train[i * self.configs.batch_size:(i + 1) * self.configs.batch_size]).float()
                if torch.cuda.is_available():
                    current_input_x = current_input_x.cuda()
                    current_input_y = current_input_y.cuda()
                self.optimizer.zero_grad()
                output = self.network(current_input_x)#.to('cuda')) #todo: network, cuda conversion, torch stack

                current_loss = self.loss(output, current_input_y.long())
                current_loss.backward()
                self.optimizer.step()

                print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, current_loss), end='\r', flush=False)

            valid_accuracy,valid_loss = self.evaluate(x_valid,y_valid,print_accuracy=False)
            results.append(dict(train_loss=current_loss,lr = self.optimizer.param_groups[0]['lr'],valid_loss=valid_loss,valid_accuracy=valid_accuracy))

            self.scheduler.step()
            print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(_epoch+1, current_loss, time()-st))

            if (_epoch+1)%self.configs.save_interval==0 or (_epoch>178 and current_loss<0.001):
                self.save(_epoch+1)

        filename = os.path.join(self.configs.save_dir,'results.pkl')
        filehandler = open(filename, 'wb')
        pickle.dump(results, filehandler)


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

    def evaluate(self, x, y,model_path=None,print_accuracy=True):
        # if model path is not specified, it takes the model currently loaded
        # else it loads the model from the model path
        preds,_ = self.predict_prob(x,model_path)
        y = torch.tensor(y)
        preds = torch.tensor(preds)
        accuracy = torch.sum(preds == y) / y.shape[0]
        loss = self.loss(preds, y.long())
        if print_accuracy:
            print('Test accuracy: {:.4f}'.format(accuracy))
        return accuracy,loss


    def predict_prob(self, x,model_path=None):
        # make sure configs have the model path, final_model path
        # also the results_directory
        if model_path:
            self.network.load_ckpt(model_path)
        self.network.eval()
        cuda_available = torch.cuda.is_available()

        pred_labels = []
        predicted_probabilities = []
        for i in tqdm(range(x.shape[0])):
            curr_x = parse_record(x[i], False).float()
            if cuda_available:
                curr_x = curr_x.cuda()
            curr_x_tensor = curr_x.view(1, 3, 32, 32)
            raw_prediction = self.network(curr_x_tensor)
            pred_labels.append(int(torch.max(raw_prediction, 1)[1]))

            probability_pred = F.softmax(raw_prediction).cpu().detach().numpy().reshape((10))
            predicted_probabilities.append(probability_pred)

        predicted_probabilities = np.array(predicted_probabilities)
        return pred_labels,predicted_probabilities


