import os
import matplotlib.pyplot as plt
from LogisticRegression import logistic_regression
from LRM import logistic_regression_multiclass
from DataReader import *
from time import time

data_dir = "../data/"
train_filename = "training.npz"
test_filename = "test.npz"
    
def visualize_features(X, y):
    '''This function is used to plot a 2-D scatter plot of training features. 

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.

    Returns:
        No return. Save the plot to 'train_features.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
    plt.clf()
    plt.scatter(X[y==1,0], X[y==1,1])#, c=y)
    plt.scatter(X[y == -1, 0], X[y == -1, 1])  # , c=y)
    #plt.legend(["class 1", "class 2"], title="classes")
    # create labels and title
    plt.xlabel('Measure of Symmetry')
    plt.ylabel('Measure of Intensity')
    plt.title('Measure of Intensity Vs Measure of Symmetry in binary classification')

    plt.savefig('../figures/train_features.jpg')
    #plt.show()

    ### END YOUR CODE


def visualize_result(X, y, W):
    '''This function is used to plot the sigmoid model after training.

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.
        W: An array of shape [n_features,].

    Returns:
        No return. Save the plot to 'train_result_sigmoid.*' and include it
        in submission.
    '''
    plt.clf()
    plt.scatter(X[y == 1, 0], X[y == 1, 1])  # , c=y)
    plt.scatter(X[y == -1, 0], X[y == -1, 1])  # , c=y)
    #plt.legend(["class 1", "class 2"], title="classes")

    symmetry_feature = X[:, 0]
    line_y_values = ( - W[0]-(W[1] * X[:, 0])) / W[2]
    plt.plot(symmetry_feature,line_y_values)
    plt.xlabel('Measure of Symmetry')
    plt.ylabel('Measure of Intensity')
    plt.title('Sigmoid result after training(binary logistic regression)')
    plt.savefig('../figures/train_result_sigmoid.jpg')
    #plt.show()


def visualize_result_multi(X, y, W):
    '''This function is used to plot the softmax model after training.

	Args:
		X: An array of shape [n_samples, 2].
		y: An array of shape [n_samples,]. Only contains 0,1,2.
		W: An array of shape [n_features, 3].
	
	Returns:
		No return. Save the plot to 'train_result_softmax.*' and include it
		in submission.
	'''
    plt.clf()
    plt.scatter(X[y == 0, 0], X[y == 0, 1])
    plt.scatter(X[y == 1, 0], X[y == 1, 1])
    plt.scatter(X[y == 2, 0], X[y == 2, 1])
    #plt.legend(["class 1", "class 2", "class 3"], title="classes")

    #plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.xlabel('Measure of Symmetry')
    plt.ylabel('Measure of Intensity')



    db1 = []
    db2 = []

    symmetry_feature = list(X[:, 0])
    symmetry_feature.sort()
    for each in symmetry_feature:
        w0, w1, w2 = (W[0], W[1], W[2])
        db1.append(np.max([((w1[0] - w0[0]) + (w1[1] - w0[1]) * each) / (w0[2] - w1[2]),
                      ((w2[0] - w0[0]) + (w2[1] - w0[1]) * each) / (w0[2] - w2[2])]))
        db2.append(np.min([((w0[0] - w1[0]) + (w0[1] - w1[1]) * each) / (w1[2] - w0[2]),
                      ((w2[0] - w1[0]) + (w2[1] - w1[1]) * each) / (w1[2] - w2[2])]))
    plt.plot(symmetry_feature, db1, '--k')
    plt.plot(symmetry_feature, db2, '--k')
    plt.title('SOFTMAX result after training(multiclass logistic regression)')
    plt.savefig('../figures/train_result_softmax.jpg')
    #plt.show()



def main():
	# ------------Data Preprocessing------------
	# Read data for training.
    
    raw_data, labels = load_data(os.path.join(data_dir, train_filename))
    raw_train, raw_valid, label_train, label_valid = train_valid_split(raw_data, labels, 2300)

    ##### Preprocess raw data to extract features
    train_X_all = prepare_X(raw_train)
    valid_X_all = prepare_X(raw_valid)
    ##### Preprocess labels for all data to 0,1,2 and return the idx for data from '1' and '2' class.
    train_y_all, train_idx = prepare_y(label_train)
    valid_y_all, val_idx = prepare_y(label_valid)  

    ####### For binary case, only use data from '1' and '2'  
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    ####### Only use the first 1350 data examples for binary training. 
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]

    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    ####### set lables to  1 and -1. Here convert label '2' to '-1' which means we treat data '1' as postitive class. 
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1
    data_shape= train_y.shape[0] 

#    # Visualize training data.
    visualize_features(train_X[:, 1:3], train_y)


   # ------------Logistic Regression Sigmoid Case------------

   ##### Check BGD, SGD, miniBGD
    logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=100)
    print('Binary logistic regression model with learning rate = 0.5 and max iterations = 100')
    print('---------------------------------------------')
    print('results of batch gradient descent')
    logisticR_classifier.fit_BGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print('Training accuracy: ',logisticR_classifier.score(train_X, train_y))
    print('validation accuracy: ',logisticR_classifier.score(valid_X, valid_y))
    print('---------------------------------------------')
    print('results of mini batch but with batch size = number of samples(basically a batch gradient descent)')
    logisticR_classifier.fit_miniBGD(train_X, train_y, data_shape)
    print(logisticR_classifier.get_params())
    print('Training accuracy: ',logisticR_classifier.score(train_X, train_y))
    print('validation accuracy: ',logisticR_classifier.score(valid_X, valid_y))
    print('---------------------------------------------')
    print('results of stochastic gradient descent')
    logisticR_classifier.fit_SGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print('Training accuracy: ',logisticR_classifier.score(train_X, train_y))
    print('validation accuracy: ',logisticR_classifier.score(valid_X, valid_y))
    print('---------------------------------------------')
    print('results of mini batch gradient descent with batch size = 1')
    logisticR_classifier.fit_miniBGD(train_X, train_y, 1)
    print(logisticR_classifier.get_params())
    print('Training accuracy: ',logisticR_classifier.score(train_X, train_y))
    print('validation accuracy: ',logisticR_classifier.score(valid_X, valid_y))
    print('---------------------------------------------')
    print('results of mini batch gradient descent with batch size  = 10')
    logisticR_classifier.fit_miniBGD(train_X, train_y, 10)
    print(logisticR_classifier.get_params())
    print('Training accuracy: ',logisticR_classifier.score(train_X, train_y))
    print('validation accuracy: ',logisticR_classifier.score(valid_X, valid_y))
    print('---------------------------------------------')
    best_logisticR = logisticR_classifier
    visualize_result(train_X[:, 1:3], train_y, best_logisticR.get_params())
    
    # Explore different hyper-parameters.
    ### YOUR CODE HERE
    ### END YOUR CODE




    # Use the 'best' model above to do testing. Note that the test data should be loaded and processed in the same way as the training data.
    ### YOUR CODE HERE
    test_data, test_labels = load_data(os.path.join(data_dir, test_filename))
    test_X_all = prepare_X(test_data)
    test_y_all,test_idx = prepare_y(test_labels)

    ####### For binary case, only use data from '1' and '2'
    test_X = test_X_all[test_idx]
    test_y = test_y_all[test_idx]
    test_y[np.where(test_y == 2)] = -1
    # Visualize the your 'best' model after training.
    ### END YOUR CODE

    ### YOUR CODE HERE

    best_logisticR = logistic_regression(learning_rate=0.2, max_iter=1000)
    print('Binary logistic regression model with learning rate = 0.2 and max iterations = 1000')
    print('Batch gradient descent is implemented')
    best_logisticR.fit_BGD(train_X, train_y)
    print('Training accuracy: ',best_logisticR.score(train_X, train_y))
    print('validation accuracy: ',best_logisticR.score(valid_X, valid_y))
    print("test accuracy (best model with given parameters is chosen):", best_logisticR.score(test_X, test_y))


    ### END YOUR CODE


    # ------------Logistic Regression Multiple-class case, let k= 3------------
    ###### Use all data from '0' '1' '2' for training
    train_X = train_X_all
    train_y = train_y_all
    valid_X = valid_X_all
    valid_y = valid_y_all

    #########  miniBGD for multiclass Logistic Regression
    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=100,  k= 3)
    print('Multi class logistic regression model with learning rate = 0.5 and max iterations = 100, with 3 classes')
    print('---------------------------------------------')
    print('results of mini batch gradient descent with batch size = 10')
    #
    logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 10)
    #
    print(logisticR_classifier_multiclass.get_params())
    print('Training accuracy: ', logisticR_classifier_multiclass.score(train_X, train_y))
    print('validation accuracy: ', logisticR_classifier_multiclass.score(valid_X, valid_y))
    print("test accuracy: ", logisticR_classifier_multiclass.score(test_X_all, test_y_all))
    visualize_result_multi(train_X[:, 1:3], train_y, logisticR_classifier_multiclass.get_params())
    # Visualize the your 'best' model after training.
    # Use the 'best' model above to do testing.
    ### YOUR CODE HERE

    print('---------------------------------------------')
    best_logistic_multi_R = logistic_regression_multiclass(learning_rate=0.1, max_iter=1000,  k= 3)
    print('Multi class logistic regression model with learning rate = 0.2 and max iterations = 1000, with 3 classes')
    print('---------------------------------------------')
    print('training with mini batch gradient descent with batch size = 100')
    best_logistic_multi_R.fit_miniBGD(train_X, train_y, 100)
    print(best_logistic_multi_R.get_params())
    print('Training accuracy: ', best_logistic_multi_R.score(train_X, train_y))
    print('validation accuracy: ', best_logistic_multi_R.score(valid_X, valid_y))
    print('------------------------------------------')
    print("test accuracy: ", best_logistic_multi_R.score(test_X_all, test_y_all))

    ### END YOUR CODE

    # ------------Connection between sigmoid and softmax------------
    ############ Now set k=2, only use data from '1' and '2' 

    #####  set labels to 0,1 for softmax classifer
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    #print(np.unique(train_y))
    train_y[np.where(train_y==2)] = 0
    valid_y[np.where(valid_y==2)] = 0
    #print(np.unique(train_y))
    
    ###### First, fit softmax classifer until convergence, and evaluate 
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE
    #logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=100, k=2)

    t1 = time()
    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.1, max_iter=10000, k=2)
    logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 10)
    print('Multiclass logistic classifier(softmax),ran to convergence with two classes.')
    print(logisticR_classifier_multiclass.get_params())
    print('Training accuracy: ', logisticR_classifier_multiclass.score(train_X, train_y))
    print('validation accuracy: ', logisticR_classifier_multiclass.score(valid_X, valid_y))
    print(logisticR_classifier_multiclass.score(train_X,train_y))
    print('time taken for training for softmax classifier till convergence  = ',time()-t1)

    ### END YOUR CODE





    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    #####       set lables to -1 and 1 for sigmoid classifer
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1   

    ###### Next, fit sigmoid classifer until convergence, and evaluate
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE



    t1 = time()
    logisticR_classifier = logistic_regression(learning_rate=0.1, max_iter=10000)
    logisticR_classifier.fit_miniBGD(train_X, train_y, 10)
    print('binary class logistic classifier(sigmoid) ran till convergence')
    print(logisticR_classifier.get_params())
    print(" training accuracy:", logisticR_classifier.score(train_X, train_y))
    print(" validation accuracy:", logisticR_classifier.score(valid_X, valid_y))
    print('time taken for training for sigmoid classifier till convergence  = ', time()-t1)
    visualize_result(train_X[:, 1:3], train_y, logisticR_classifier.get_params())
    ### END YOUR CODE

    ################Compare and report the observations/prediction accuracy


    '''
    Explore the training of these two classifiers and monitor the graidents/weights for each step. 
    Hint: First, set two learning rates the same, check the graidents/weights for the first batch in the first epoch. What are the relationships between these two models? 
    Then, for what leaning rates, we can obtain w_1-w_2= w for all training steps so that these two models are equivalent for each training step. 
    '''
    ### YOUR CODE HERE
    sigmoid = logistic_regression(learning_rate=0.2, max_iter=1)
    sigmoid.fit_miniBGD(train_X, train_y, 20)
    print("sigmoid classifier weights:\n", sigmoid.get_params())

    train_y[np.where(train_y == -1)] = 0
    softmax = logistic_regression_multiclass(learning_rate=0.1, max_iter=1, k=2)
    softmax.fit_miniBGD(train_X, train_y, 20)
    print("softmax classifier weights:\n", softmax.get_params())



    ### END YOUR CODE

    # ------------End------------
    

if __name__ == '__main__':
	main()
    
    
