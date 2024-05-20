import numpy as np


def five_cube():
    graph = []  # generate 5-hypercube with 32 nodes where each node is connected to 5 neighbors

    for i in range(32):
        graph.append((i, (i + 1) % 32))
        graph.append((i, (i + 8) % 32))
        graph.append((i, (i - 1) % 32))
        graph.append((i, (i - 8) % 32))

    for i in range(32):
        graph.append((i, (i + 16) % 32))

    graph_mat = np.eye(32)
    for i, j in graph:
        graph_mat[i, j] = 1
    # print(graph_mat)
    graph_mat /= 6.
    return graph_mat


def two2six_cycle():
    graph = []  # generate 5-hypercube with 32 nodes where each node is connected to 5 neighbors

    for i in range(64):
        graph.append((i, (i + 1) % 64))
        graph.append((i, (i - 1) % 64))
    # print(graph)
    graph_mat = np.eye(64)
    for i, j in graph:
        graph_mat[i, j] = 1
    # print(graph_mat)
    graph_mat /= 3.
    return graph_mat


def hamming_dist(x, y):
    tmp = x ^ y
    tmp = bin(tmp)[2:]
    # print(bin(x))
    # print(bin(y))
    # print([int(i) for i in tmp])
    dist = np.sum([int(i) for i in tmp])
    return dist


def hyper_cube(exp_num):
    # print(hamming_dist(15,9))
    # print(bin(3^2))
    graph = []
    n = 2 ** exp_num
    for i in range(n):
        for j in range(n):
            dist = hamming_dist(i, j)
            if (dist <= 1):
                graph.append((i, j))
    graph_mat = np.eye(n)
    for i, j in graph:
        graph_mat[i, j] = 1
    print(graph_mat)
    graph_mat /= (exp_num + 1)
    return graph_mat


def two2three_cycle():
    graph = []  # generate 5-hypercube with 32 nodes where each node is connected to 5 neighbors

    for i in range(8):
        graph.append((i, (i + 1) % 8))
        graph.append((i, (i - 1) % 8))
    # print(graph)
    graph_mat = np.eye(8)
    for i, j in graph:
        graph_mat[i, j] = 1
    # print(graph_mat)
    graph_mat /= 3.
    return graph_mat


def n_cycle(n):
    graph = []  # generate 5-hypercube with 32 nodes where each node is connected to 5 neighbors

    for i in range(n):
        graph.append((i, (i + 1) % n))
        graph.append((i, (i - 1) % n))
    # print(graph)
    graph_mat = np.eye(n)
    for i, j in graph:
        graph_mat[i, j] = 1
    # print(graph_mat)
    graph_mat /= 3.
    return graph_mat


if __name__ == '__main__':
    fivecube = five_cube()
    # np.savetxt("fivecube.txt", fivecube)
    col_sum = fivecube.sum(axis=0)
    row_sum = fivecube.sum(axis=1)
    print(col_sum)
    print(row_sum)
