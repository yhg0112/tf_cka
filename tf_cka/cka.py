import numpy as np
import tensorflow as tf
import torch


def linear_kernel(X):
    return np.dot(X, X.transpose())


def centering(X):
    if not np.allclose(X, X.T):
        raise ValueError('Input must be a symmetric matrix.')

    means = np.mean(X, 0)
    means -= np.mean(means) / 2

    centered_X = X - means[:, None]
    centered_X -= means[None, :]

    return centered_X


def hsic(X, Y, kernel=linear_kernel):
    gram_X = kernel(X)
    gram_Y = kernel(Y)

    centered_gram_X = centering(gram_X)
    centered_gram_Y = centering(gram_Y)

    scaled_hsic = np.dot(np.ravel(centered_gram_X), np.ravel(centered_gram_Y))

    return scaled_hsic


def cka(X, Y, kernel=linear_kernel):
    gram_X = kernel(X)
    gram_Y = kernel(Y)

    centered_gram_X = centering(gram_X)
    centered_gram_Y = centering(gram_Y)

    scaled_hsic = np.dot(np.ravel(centered_gram_X), np.ravel(centered_gram_Y))

    norm_X = np.linalg.norm(centered_gram_X)
    norm_Y = np.linalg.norm(centered_gram_Y)

    return scaled_hsic / (norm_X * norm_Y)


def linear_kernel_tf(X):
    return tf.matmul(X, X, transpose_b=True)


def centering_tf(X):
    means = tf.reduce_mean(X, axis=0)
    means -= tf.reduce_mean(means) / 2.

    centered_X = X - means[:, None]
    centered_X -= means[None, :]

    return centered_X


def hsic_tf(X, Y, kernel=linear_kernel_tf):
    gram_X = kernel(X)
    gram_Y = kernel(Y)

    centered_gram_X = centering_tf(gram_X)
    centered_gram_Y = centering_tf(gram_Y)

    scaled_hsic = tf.tensordot(tf.reshape(centered_gram_X, shape=[-1]), tf.reshape(centered_gram_Y, shape=[-1]), axes=1)

    return scaled_hsic


def cka_tf(X, Y, kernel=linear_kernel_tf):
    gram_X = kernel(X)
    gram_Y = kernel(Y)

    centered_gram_X = centering_tf(gram_X)
    centered_gram_Y = centering_tf(gram_Y)

    scaled_hsic = tf.tensordot(tf.reshape(centered_gram_X, shape=[-1]), tf.reshape(centered_gram_Y, shape=[-1]), axes=1)

    norm_X = tf.norm(centered_gram_X)
    norm_Y = tf.norm(centered_gram_Y)

    return scaled_hsic / (norm_X * norm_Y)

def linear_kernel_torch(X):
    #size_X = X.size()
    #dim_ind = list(range(len(size_X)))
    # permute dim_ind
    #dim_ind[-2] = dim_ind[-1] 
    #dim_ind[-1] = len(size_X) -2 
    return torch.matmul(X, X.t())

def centering_torch(X):
    if torch.__version__ < '1.2':  ## python 2, torch 1.1
        means = torch.mean(X, [1])
    else:
        means = torch.mean(X, axis=0)
        means -= torch.mean(means) / 2.

    centered_X = X - means[:, None]
    centered_X -= means[None, :]

    return centered_X

def hsic_torch(X, Y, kernel=linear_kernel_torch):
    gram_X = kernel(X)
    gram_Y = kernel(Y)

    centered_gram_X = centering_torch(gram_X)
    centered_gram_Y = centering_torch(gram_Y)

    scaled_hsic = torch.dot(torch.flatten(centered_gram_X), torch.flatten(centered_gram_Y))
    #scaled_hsic = torch.dot(torch.reshape(centered_gram_X, shape=[-1]), torch.reshape(centered_gram_Y, shape=[-1]))

    return scaled_hsic

def cka_torch(X, Y, kernel=linear_kernel_torch):
    gram_X = kernel(X)
    gram_Y = kernel(Y)

    centered_gram_X = centering_torch(gram_X)
    centered_gram_Y = centering_torch(gram_Y)

    scaled_hsic = torch.dot(torch.flatten(centered_gram_X), torch.flatten(centered_gram_Y))    
    #scaled_hsic = torch.dot(torch.reshape(centered_gram_X, shape=[-1]), torch.reshape(centered_gram_Y, shape=[-1]))

    norm_X = torch.norm(centered_gram_X)
    norm_Y = torch.norm(centered_gram_Y)

    return scaled_hsic / (norm_X * norm_Y)
    