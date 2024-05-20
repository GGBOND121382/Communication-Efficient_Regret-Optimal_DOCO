import copy

import numpy as np
import math
# from config_save_load import conf_load
# from libsvm_data_load import load_libsvm_data
from sklearn.preprocessing import MaxAbsScaler

import sys
sys.path.append('../')

from data.libsvm_data_load import *


def diag_matrix_sqr(S):
    S_diag = np.diagonal(S)
    S_diag_sqr = S_diag ** 2
    S_sqr = np.diag(S_diag_sqr)
    return S_sqr


def diag_matrix_inv(S):
    S_diag = np.diagonal(S)
    S_diag_inv = 1 / S_diag
    S_inv = np.diag(S_diag_inv)
    return S_inv

def diag_matrix_mul(A, B):
    A_diag = np.diagonal(A)
    B_diag = np.diagonal(B)
    AB_diag = A_diag * B_diag
    AB = np.diag(AB_diag)
    return AB

def A_inv_mul_b(A, b):
    ret_val = np.linalg.solve(A, b)
    return ret_val


def Q_sqr(x, Q):
    tmp = np.dot(x.T, Q)
    tmp = np.dot(tmp, x)
    return tmp


def Q_inv_sqr(x, Q):
    tmp = np.linalg.solve(Q, x)
    tmp = np.dot(x.T, tmp)
    return tmp


def cutting_oracle_l2(x):
    return x


def CP_Vaidya_stoc_dist(A, b, x, O_grad, O_loss, O_cut, data_collection, target_collection, batchsize, num_clients, mu,
                        alpha_step5=0.5, eps=0.04, gamma=0.01, radius=10, is_l2_norm=True):
    time_horizon, n_features = data_collection[0].shape
    iteration = int(time_horizon / batchsize)
    print("t_i:", time_horizon)
    print("iteration:", iteration)
    print("batch size:", batchsize)
    min_loss = None
    ret_x = copy.deepcopy(x)
    # for t in range(iteration):
    t = 0
    while t < iteration:
        print("t:", t)
        sigma = sigma_func(A, b, x)
        sigma_min = np.min(sigma)
        if (sigma_min >= eps):  ##Step3
            if (is_l2_norm and np.linalg.norm(x) > radius):
                print("x not in C")
                w = -O_cut(x)
            elif (not is_l2_norm and np.max(np.abs(x)) > radius):
                assert False
            else:
                startTime = t * batchsize
                if (t < iteration - 1):
                    endTime = (t + 1) * batchsize - mu
                else:
                    endTime = max(time_horizon - mu, startTime)
                X_train, y_train = load_data_interval_dist(data_collection, target_collection, n_features, startTime,
                                                           endTime, -1, num_clients)
                w = -O_grad(x, X_train, y_train)
                loss = O_loss(x, X_train, y_train)
                print("loss:", loss)

                if (min_loss != None):
                    if (loss < min_loss):
                        ret_x = copy.deepcopy(x)
                        print(ret_x)
                        min_loss = loss
                else:
                    min_loss = loss
                t += 1

            bkplus1 = np.dot(w, x)
            sk = np.dot(A, x) - b
            Sk = np.diag(sk)
            # tmp = Sk ** 2
            tmp = diag_matrix_sqr(Sk)
            # tmp = np.linalg.inv(tmp)
            tmp = diag_matrix_inv(tmp)
            ATS_neg2A = Q_sqr(A, tmp)  # A.T*S(x)**(-2)*A
            # ATS_neg2A_inv = np.linalg.inv(ATS_neg2A)
            # newton_step = np.dot(ATS_neg2A_inv, w)
            newton_step = np.linalg.solve(ATS_neg2A, w)
            alpha_step3 = 0.3 / np.sqrt(np.dot(w.T, newton_step))
            x = x + alpha_step3 * newton_step  # move the point x
            #########################################################
            A = np.vstack((A, w))
            b = np.hstack((b, bkplus1))  # add a constraint
            #########################################################
            # print("[1] add a constraint..........................................")
        else:  ##Step4: I wonder whether it worths to neglect the evaluated gradients !!!
            sigma_list = sigma.tolist()
            ind = sigma_list.index(sigma_min)
            A = np.delete(A, ind, axis=0)
            b = np.delete(b, ind)  # remove a constraint
            # print("[2] remove a constraint.......................................")
        x = volume_center(A, b, x)
    # print(ret_x)
    return ret_x


def volume_center(A, b, x, gamma=0.01, alpha=0.5):
    grad = g_func(A, b, x)
    mu = mu_hat_func(A, b, x)
    Q_mat = Q_func(A, b, x)
    # Q_inv = np.linalg.inv(Q_mat)
    # judge = mu * Q_sqr(grad, Q_inv)
    Q_inv_grad = np.linalg.solve(Q_mat, grad)
    gT_Q_inv_g = np.dot(grad.T, Q_inv_grad)
    judge = mu * gT_Q_inv_g
    while (judge >= gamma):
        x = x - alpha * Q_inv_grad
        grad = g_func(A, b, x)
        mu = mu_hat_func(A, b, x)
        Q_mat = Q_func(A, b, x)
        # Q_inv = np.linalg.inv(Q_mat)
        # judge = mu * Q_sqr(grad, Q_inv)
        Q_inv_grad = np.linalg.solve(Q_mat, grad)
        gT_Q_inv_g = np.dot(grad.T, Q_inv_grad)
        judge = mu * gT_Q_inv_g
    return x


def Vk_func():
    pass


def V_func(A, b, x):
    sx = np.dot(A, x) - b
    Sx = np.diag(sx)

    # tmp = Sx ** 2
    # tmp = np.linalg.inv(tmp)
    # tmp = Q_sqr(A, tmp)  # A.T*S(x)**(-2)*A

    tmp = diag_matrix_sqr(Sx)
    tmp = diag_matrix_inv(tmp)
    tmp = Q_sqr(A, tmp)     # A.T*S(x)**(-2)*A

    tmp_ldet = np.linalg.det(tmp)
    tmp_ldet = np.log(tmp_ldet)  # ln(det(tmp))
    return 0.5 * tmp_ldet


def P_func(A, b, x):
    sx = np.dot(A, x) - b
    Sx = np.diag(sx)

    # tmp = Sx ** 2
    # tmp = np.linalg.inv(tmp)
    tmp = diag_matrix_sqr(Sx)
    tmp = diag_matrix_inv(tmp)
    ATS_neg2A = Q_sqr(A, tmp)  # A.T*S(x)**(-2)*A

    S_inv = diag_matrix_inv(Sx)
    ATS_inv = np.dot(A.T, S_inv)

    # ATS_neg2A_inv = np.linalg.inv(ATS_neg2A)
    # P_mat = Q_sqr(ATS_inv, ATS_neg2A_inv)
    P_mat = Q_inv_sqr(ATS_inv, ATS_neg2A)
    return P_mat


def sigma_func(A, b, x):
    P_mat = P_func(A, b, x)
    sigma = np.diagonal(P_mat)
    return sigma


def Sigma_func(A, b, x):
    sigma = sigma_func(A, b, x)
    Sigma = np.diag(sigma)
    return Sigma


def Q_func(A, b, x):
    sx = np.dot(A, x) - b
    Sx = np.diag(sx)
    # tmp = Sx ** 2
    tmp = diag_matrix_sqr(Sx)
    # tmp = np.linalg.inv(tmp)  # S**(-2))
    tmp = diag_matrix_inv(tmp)  # S**(-2))
    Sigma = Sigma_func(A, b, x)
    # tmp = np.dot(tmp, Sigma)  # S**(-2) * Sigma
    tmp = diag_matrix_mul(tmp, Sigma)
    Q_mat = Q_sqr(A, tmp)
    return Q_mat


def g_func(A, b, x):
    sx = np.dot(A, x) - b
    Sx = np.diag(sx)
    # S_inv = np.linalg.inv(Sx)
    S_inv = diag_matrix_inv(Sx)
    ATS_inv = np.dot(A.T, S_inv)
    sigma = sigma_func(A, b, x)
    grad = - np.dot(ATS_inv, sigma)
    return grad


def mu_hat_func(A, b, x):
    sigma = sigma_func(A, b, x)
    # print("sigma:", sigma)
    if (sigma.shape[0] == 0):
        assert False
    sigma_min = np.min(sigma)
    m = A.shape[0]
    tmp1 = 2 * np.sqrt(sigma_min) - sigma_min
    mu = np.sqrt((1. + np.sqrt(m)) / 2.)
    if (tmp1 > 0):
        tmp1 = tmp1 ** (-0.5)
        if (tmp1 < mu):
            mu = tmp1
    return mu


def Hessian(A, b, x):
    pass


if __name__ == "__main__":
    # an example of cutting plane: min x^2 s.t. ||x||_{\infty}\leq 1
    d = 100
    radius = 10
    A = np.vstack((np.eye(d), -np.eye(d)))
    b = - np.ones(2 * d) * radius
    x = np.zeros(d)
    T = 40
    xstar = np.zeros(d)
    xstar[0] = 2
    xstar[2] = 5
    O_grad = lambda x,X,Y: 2 * (x - xstar)
    O_loss = lambda x,X,Y: np.linalg.norm(x - xstar) ** 2
    O_cut = lambda x: x
    # x = CP_Vaidya(A, b, x, T, O_grad, cutting_oracle_l2, radius=1, is_l2_norm=True, eps=0)
    data_collection = [np.zeros((100, 1)), np.zeros((100, 1))]
    target_collection = [np.zeros((100, 1)), np.zeros((100, 1))]
    x = CP_Vaidya_stoc_dist(A, b, x, O_grad, O_loss, O_cut, data_collection, target_collection, 3, 2, 2,
                            alpha_step5=0.5, eps=0, radius=radius, is_l2_norm=True)
    print("resulting x:", x)