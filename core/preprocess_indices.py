# given a sequence of indices (mu)
import numpy as np


def compute_contexts(idx_seq, kmax, left_padding=-1, right_padding=-2):
    M = len(idx_seq)
    ctx_length = 2 * kmax + 1
    ctx = np.zeros((M, ctx_length))
    padded_seq = np.concatenate([np.array([left_padding] * kmax), idx_seq, np.array([right_padding] * kmax)])
    assert len(padded_seq) == kmax * 2 + M
    for mu, idx in enumerate(idx_seq):
        ctx[mu, :] = padded_seq[mu:mu + ctx_length]
    assert ctx.shape == (M, ctx_length)
    return ctx


def compute_context_indices(contexts, kmax=0):
    M, ctx_length = contexts.shape
    kmax = kmax or (ctx_length - 1) // 2
    result = np.zeros((M, 3, 2 * kmax))
    for mu in range(M):
        for k in range(kmax):
            s_0 = contexts[mu, kmax]
            s_k = contexts[mu, kmax + k + 1]
            s_mk = contexts[mu, kmax - k - 1]
            result[mu, 0, 2 * k: 2 * (k + 1)] = k
            result[mu, 1, 2 * k] = s_mk
            result[mu, 1, 2 * k + 1] = s_0
            result[mu, 2, 2 * k] = s_0
            result[mu, 2, 2 * k + 1] = s_k
    return result

def compute_normalization_indices():
    # TODO



if __name__ == '__main__':
    seq = np.array([1, 2, 3, 4, 5])
    ctx = compute_contexts(seq, 5)
    print(ctx)
    print(compute_context_indices(ctx))
