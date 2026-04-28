def prange(): ...
def srange(): ...
tile_k = ...

for n in prange(4096, tile=128):

    # matmul X, [WQ, WK, WV] + acc
    for k in srange(4096, tile=tile_k):
        tmp1 = load(X[:, k]) @ load(WQ, WK, WV[k, n])
        tmp2 = load(Q1, K1, V1)[:, n] * 1
        tmp3 = tmp2 + tmp1 # V1 is accumulator
        store(Q1, K1, V1)

...