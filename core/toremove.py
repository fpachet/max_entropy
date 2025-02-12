import time

import numpy as np

if __name__ == '__main__':
    K = 10
    q = 100

    t0 = time.perf_counter_ns()

    for i in range(1000):

        Z = np.array(np.arange(0, K * q * q)).reshape((K, q, q))
    # print(np.array2string(Z))
        Z = Z / np.max(Z)
        indices = np.where(np.mod(Z, 7) == 0)
        res = np.exp(np.copy(Z)[indices]).sum()

    print((time.perf_counter_ns() - t0) / 1_000_000)
    #
    # context = np.array([0, 0, 2, 4, 5])
    # indices_context = (np.array([0, 0, 1, 1]),
    #                    np.array([0, 2, 0, 2]),
    #                    np.array([2, 4, 2, 5]))
    # print(indices_context)
    # print(Z[indices_context])
    # print(Z[indices_context].sum())

