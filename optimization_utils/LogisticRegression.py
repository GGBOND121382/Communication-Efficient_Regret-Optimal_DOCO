import copy

import numpy as np
import math
# from config_save_load import conf_load
# from libsvm_data_load import load_libsvm_data
from sklearn.preprocessing import MaxAbsScaler


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def class_err_logReg(x, X, y, is_sum=False):
    h_x = X.dot(x)
    y_pred = sigmoid(h_x)

    y_pred = np.reshape(y_pred, (-1, 1))

    m_samples = len(y_pred)

    h_x = np.reshape(h_x, (-1, 1))
    # print("y_shape:", y.shape)
    # print("h_x_shape:", h_x.shape)
    if (is_sum):
        class_err = m_samples - np.sum(np.around(y_pred) == y)
    else:
        class_err = 1 - np.mean(np.around(y_pred) == y)
    return class_err


def grad_logReg(x, X, y, is_sum=False):
    h_x = X.dot(x)
    y_pred = sigmoid(h_x)
    y_pred = np.reshape(y_pred, (-1, 1))
    if (is_sum):
        w_grad = np.sum(X * (y_pred - y), axis=0)
    else:
        w_grad = np.mean(X * (y_pred - y), axis=0)
    return w_grad


def loss_logReg(x, X, y, is_sum=False):
    h_x = X.dot(x)
    y_pred = sigmoid(h_x)

    y_pred = np.reshape(y_pred, (-1, 1))

    h_x = np.reshape(h_x, (-1, 1))
    # print("y_shape:", y.shape)
    # print("h_x_shape:", h_x.shape)
    if (is_sum):
        loss = np.sum(y * np.log(1 + np.exp(-h_x)) + (1 - y) * np.log(1 + np.exp(h_x)))
    else:
        loss = np.mean(y * np.log(1 + np.exp(-h_x)) + (1 - y) * np.log(1 + np.exp(h_x)))
    return loss


def gradient_descent(x, grad, learning_rate, radius, is_l2_norm=False):
    x = x - learning_rate * grad
    x = projection(x, radius, is_l2_norm)
    return x


def projection(x, radius, is_l2_norm=False):
    if (is_l2_norm):
        if (np.linalg.norm(x) > radius):
            x = radius * x / np.linalg.norm(x)
    else:
        for k in range(len(x)):  # Projection
            if x[k] > radius:
                x[k] = radius
            if x[k] < -radius:
                x[k] = -radius
    return x