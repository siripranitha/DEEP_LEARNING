import torch
from helper import load_data
from solution import PCA, AE, frobeniu_norm_error
import numpy as np
import os


def test_pca(A, p):
    pca = PCA(A, p)
    Ap, G = pca.get_reduced()
    A_re = pca.reconstruction(Ap)
    error = frobeniu_norm_error(A, A_re)
    print('PCA-Reconstruction error for {k} components is'.format(k=p), error)
    return G

def test_ae(A, p,batch_size=128, max_epoch=300):
    model = AE(d_hidden_rep=p)
    model.train(A, A, batch_size, max_epoch)
    A_re = model.reconstruction(A)
    final_w = model.get_params()
    error = frobeniu_norm_error(A, A_re)
    print('AE-Reconstruction error for {k}-dimensional hidden representation is'.format(k=p), error)
    return final_w,error

if __name__ == '__main__':
    dataloc = "../data/USPS.mat"
    A = load_data(dataloc)
    A = A.T
    ## Normalize A
    A = A/A.max()

    ### YOUR CODE HERE
    # Note: You are free to modify your code here for debugging and justifying your ideas for 5(f)
    ps = [64]#, 64, 128]#[64]#,64,128] #, 64, 128]#[50, 100, 150]
    e1 = []
    e2 = []
    batch_size = [64,128]#[32]
    max_epochs = [300,500]#,700,1000]
    result = []

    for each_bs in batch_size:
        for each_max_ep in max_epochs:
            final_w, error = test_ae(A, 64,each_bs,each_max_ep)
            result.append(f'Max epochs = {each_max_ep}; batch size = {each_bs}; AE-Reconstruction error = {error} ')
    print(result)
    # for p in ps:
    #     #G = test_pca(A, p)
    #     final_w,error = test_ae(A, p)

        #e1.append(frobeniu_norm_error(G,final_w))
        #e2.append(frobeniu_norm_error(G.T @G,final_w.T@final_w))
    ### END YOUR CODE

    #print(f'direct comparision : {e1}')
    #print(f'Indirect comparision : {e2}')
