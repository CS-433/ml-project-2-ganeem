from scipy.sparse import *
import numpy as np
import pickle
import random


def main():
    print("loading cooc")
    with open("cooc.pkl", "rb") as f:
        cooc = pickle.load(f)
    print(f"{cooc.nnz} non-zero elements")

    nmax = 100
    print("using nmax =",nmax, "(number of closest words to consider), cooc.max() = ",cooc.max(),"(number of times the most common word appears)")

    print("initializing embeddings")
    embedding_dim = 20 # word embedding dimension
    xs = np.random.normal(size=(cooc.shape[0], embedding_dim))
    ys = np.random.normal(size=(cooc.shape[1], embedding_dim))

    eta = 0.001 # learning rate
    alpha = 3 / 4 # smoothing parameter

    epochs = 10

    for epoch in range(epochs):
        print(f"epoch {epoch}")
        for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):
            logn = np.log(n)
            fn = min(1.0, (n / nmax) ** alpha)
            x, y = xs[ix, :], ys[jy, :]
            scale = 2 * eta * fn * (logn - np.dot(x, y))
            xs[ix, :] += scale * y
            ys[jy, :] += scale * x
    np.save("embeddings", xs)

if __name__ == "__main__":
    main()