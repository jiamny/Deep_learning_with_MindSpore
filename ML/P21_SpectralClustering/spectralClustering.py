"""
Reference: https://en.wikipedia.org/wiki/Spectral_clustering
Blog Post: https://towardsdatascience.com/spectral-clustering-aba2640c0d5b
"""
import mindspore as ms
import numpy as np
from mindspore import context, Tensor, nn

from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans

ms.set_seed(42)
context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

def SpectralClustering(X, K=8, adj=True, metric='euclidean', sim_graph='fully_connect', sigma=1, knn=10, epsilon=0.5, normalized=1):
    """
    :param X: Input tensor
    :param K: cluster to look out for using KMeans
    :param adj: Adjacency Matrix
    :param metric:
    :param sim_graph: Technique to create edges between nodes in graph.
    :param sigma: Parameter for RBF kernel
    :param knn: To connect with 10 nearest neighors with edges
    :param epsilon:Parameter for finding edges
    :param normalized:
    :return:
    """

    # To convert our adjacency matrix as connected graph we can use technique like KNN.

    if not adj:
        adjacency_matrix = squareform(X, metric=metric)
    else:
        adjacency_matrix = X

    if sim_graph == 'fully_connect':
        adjacency_matrix = Tensor(adjacency_matrix)
        w = ms.ops.exp(-adjacency_matrix/ (2 * sigma))

    elif sim_graph =='eps_neighbor':
        adjacency_matrix = Tensor(adjacency_matrix)
        w = (adjacency_matrix <= epsilon).astype(ms.float32)
    elif sim_graph == 'knn':
        adjacency_matrix = Tensor(adjacency_matrix)
        w = ms.ops.zeros(adjacency_matrix.shape)
        adjacency_sort = ms.ops.argsort(adjacency_matrix, axis=1)
        for i in range(adjacency_sort.shape[0]):
            w[i, adjacency_sort[i, :][:(knn+1)]] = 1
    elif sim_graph == 'mutual_knn':
        adjacency_matrix = Tensor(adjacency_matrix)
        w1 = ms.ops.zeros(adjacency_matrix.shape)
        adjacency_sort = ms.ops.argsort(adjacency_matrix, axis=1)

        for i in range(adjacency_matrix.shape[0]):
            for j in adjacency_sort[i, :][:(knn+1)]:
                if i==j:
                    w1[i, i] = 1
                elif w1[i, j] == 0 and w1[j, i]==0:
                    w1[i, j] = 0.5
                else:
                    w1[i, j] = w1[j, i] = 1
        #w = w1[w1>0.5].astype(ms.float32).copy()
        w = (w1 > 0.5).astype(ms.float32)
    else:
        raise ValueError("The 'sim_graph' argument should be one of the strings, 'fully_connect', 'eps_neighbor', 'knn', or 'mutual_knn'!")

    #Degree Matrix
    D = ms.ops.diag(ms.ops.sum(w, dim=1))

    #Graph Laplacian
    L = D - w

    # Finding eigen Value of Graph Laplacian Matrix,
    """
    The eigenvalues of the Laplacian indicated that there were four clusters.
    The vectors associated with those eigenvalues contain information on how to segment the nodes.
    """
    if normalized == 1:
        D_INV = ms.ops.diag(1/ms.numpy.diag(D)) # mindspore.numpy.diag 用于提取对角线组成的一维张量

        lambdas, V = ms.ops.eig(ms.ops.mm(D_INV, L))
        lambdas = ms.ops.real(lambdas)
        V = ms.ops.real(V)
        ind = ms.ops.argsort(ms.ops.norm(ms.ops.reshape(lambdas, (1, len(lambdas))), dim=0))
        V_K = V[:, ind[:K]]

    elif normalized == 2:
        D_INV_SQRT = ms.ops.diag(1/ms.ops.sqrt(ms.ops.diag(D)))
        lambdas, V = ms.ops.eig(ms.ops.matmul(ms.ops.matmul(D_INV_SQRT, L), D_INV_SQRT))
        lambdas = ms.ops.real(lambdas)
        V = ms.ops.real(V)
        ind = ms.ops.argsort(ms.ops.norm(ms.ops.reshape(lambdas, (1, len(lambdas))), dim=0))
        V_K = ms.ops.real(V[:, ind[:,K]])
        if any(V_K.sum(dim=1) == 0):
            raise ValueError("Can't normalize the matrix with the first K eigenvectors as columns! Perhaps the \
                             number of clusters K or the number of neighbors in k-NN is too small.")
        V_K = V_K/ms.ops.reshape(ms.ops.norm(V_K, dim=1), (V_K.shape[0], 1))
    else:
        lambdas, V = ms.ops.eig(L)
        lambdas = ms.ops.real(lambdas)
        V = ms.ops.real(V)
        ind = ms.ops.argsort(ms.ops.norm(ms.ops.reshape(lambdas, (1, len(lambdas))), dim=0))
        V_K = ms.ops.real(V[:, ind[:K]])

    # KMeans is used for assigning the labels to the clusters.
    kmeans = KMeans(n_clusters=K, init='k-means++', random_state=0).fit(V_K)
    return kmeans

if __name__ == '__main__':
    moon_data, moon_labels = make_moons(100, noise=0.05)
    moon_data = Tensor(moon_data, dtype=ms.float32)
    moon_labels = Tensor(moon_labels, dtype=ms.int32)
    # Compute the adjacency matrix, Similarity Matrix.
    padj = pdist(moon_data.asnumpy(), metric='euclidean')
    Adj_mat = squareform(padj)

    # Spectral clustering...
    #sim_graph = 'mutual_knn'  # 'eps_neighbor' 'fully_connect' 'knn'
    sim_graph = 'fully_connect'
    spec_re1 = SpectralClustering(Adj_mat, K=2, sim_graph=sim_graph, sigma=0.01, normalized=1)
    spec_re2 = SpectralClustering(Adj_mat, K=2, sim_graph='knn', knn=10, normalized=1)

    # Often need to change figsize when doing subplots
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(x=moon_data[:, 0], y=moon_data[:, 1], c=spec_re1.labels_, s=2)
    plt.colorbar()
    plt.title( sim_graph + ' graph with RBF kernel ($\sigma=0.01$)')

    plt.subplot(1, 2, 2)
    plt.scatter(x=moon_data[:, 0], y=moon_data[:, 1], c=spec_re2.labels_, s=2)
    plt.colorbar()
    plt.title('$k$-Nearest Neighbor graphs ($k=10$)')
    plt.suptitle('Spectral Clustering', y=-0.01)

    # Automatrically adjust padding between subpots
    plt.tight_layout()
    plt.show()

    exit(0)



