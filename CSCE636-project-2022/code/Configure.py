# Below configures are examples, 
# you can modify them as you wish.

### YOUR CODE HERE


import argparse
import os


def configure():
	parser = argparse.ArgumentParser()
	parser.add_argument("--resnet_size", type=int, default=3,
						help='n: the size of ResNet-(6n+2) v1 or ResNet-(9n+2) v2')
	parser.add_argument("--batch_size", type=int, default=128, help='training batch size')
	parser.add_argument("--num_classes", type=int, default=10, help='number of classes')
	parser.add_argument("--save_interval", type=int, default=10,
						help='save the checkpoint when epoch MOD save_interval == 0')

	parser.add_argument("--first_num_filters", type=int, default=16, help='number of classes')
	parser.add_argument("--weight_decay", type=float, default=2e-4, help='weight decay rate')
	parser.add_argument("--learning_rate",type=float,default=0.01,help='learning rate')
	parser.add_argument("--momentum", type=float, default=0.9, help='momentum')
	parser.add_argument("--width", type=int, default=3, help="width of the resnet")
	parser.add_argument("--max_epoch",type=int,default=200,help="epoch count")

	DATA_DIR = "../../ResNet/cifar-10-batches-py"
	#print(os.listdir("../../"))
	SAVE_DIR = "../save"
	#parser.add_argument("--modeldir", type=str, default='model_v1', help='model directory')
	parser.add_argument("--mode",default='train', help="train, test or predict")
	parser.add_argument("--data_dir", default=DATA_DIR,help="path to the data")
	parser.add_argument("--save_dir", default=SAVE_DIR,help="path to save the results")
	MODEL_PATH = None
	parser.add_argument("--final_model_path",default=MODEL_PATH)
	return parser.parse_args()



model_configs = {
	"name": 'MyModel',
	"save_dir": '../saved_models/',
	"depth": 2,
	# ...
}



training_configs = {
	"learning_rate": 0.01,
	# ...
}

### END CODE HERE