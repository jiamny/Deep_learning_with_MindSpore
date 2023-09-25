import mindspore as ms
from mindspore import dtype as mstype
from mindspore import Tensor
import scipy
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mindspore.common import set_seed
from mindspore import context

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

class KMeans:
    def __init__(self, X, k, iterations):
        """
        :param X: input tensor
        :param k: Number of clusters
        :variable samples: Number of samples
        :variable features: Number of features
        """
        self.k = k
        self.max_iterations = iterations
        self.samples = X.shape[0]
        self.features = X.shape[1]
        self.KMeans_Centroids = []

    # def initialize_centroid(self, X):
    #     return X[torch.randint(X.shape[0], (self.k,))]

    def initialize_centroid(self, X, K):
        """
        Initialization Technique is KMeans++. Thanks to stackoverflow.
        https://stackoverflow.com/questions/5466323/how-could-one-implement-the-k-means-algorithm
        :param X: Input Tensor
        :param K: Number of clusters to build
        :return: Selection of three centroid vector from X
        """
        X = X.asnumpy()
        I = [0]
        C = [X[0]]
        for k in range(1, K):
            D2 = np.array([min([np.inner(c - x, c - x) for c in C]) for x in X])

            probs = D2 / D2.sum()
            cumprobs = np.cumsum(probs)
            r = np.random.rand(1)

            print('cumprobs ', cumprobs)
            for j, p in enumerate(cumprobs):
                if r < p:
                    i = j
                    break
            print('i ', i)
            I.append(i)
        print('I: ', I)
        return ms.Tensor(X[I])

    def distance(self, sample, centroid, dim=None, default="euclidean"):
        if default == "euclidean":
            return ms.ops.norm(sample - centroid, 2, 0)
        elif default == "manhattan":
            return ms.ops.sum(ms.ops.abs(sample  - centroid))
        elif default == "cosine":
            return ms.ops.sum(sample * centroid) / (ms.ops.norm(sample) * ms.ops.norm(centroid))
        else:
            raise ValueError("Unknown similarity distance type")

    def closest_centroid(self, sample, centroids):
        """
        :param sample: sample whose distance from centroid is to be measured
        :param centroids: all the centroids of all the clusters
        :return: centroid's index is passed for each sample
        """
        closest = None
        min_distance = float('inf')
        for idx, centroid in enumerate(centroids):
            print(sample)
            print(centroid)
            distance = self.distance(sample, centroid, default="euclidean")
            print(distance)
            if distance < min_distance:
                closest = idx
                min_distance = distance

        return closest

    def create_clusters(self, centroids, X):
        """
        :param centroids: Centroids of all clusters
        :param X: Input tensor
        :return: Assigning each sample to a cluster.
        """
        n_samples = X.shape[0]
        k_clusters = [[] for _ in range(self.k)]
        for idx, sample in enumerate(X):
            centroid_index = self.closest_centroid(sample, centroids)
            k_clusters[centroid_index].append(idx)

        return k_clusters

    def update_centroids(self, clusters, X):
        """
        :return: Updating centroids after each iteration.
        """
        centroids = ms.ops.zeros((self.k, self.features))
        for idx, cluster in enumerate(clusters):
            centroid = ms.ops.mean(X[cluster], axis=0)
            centroids[idx] = centroid

        return centroids

    def label_clusters(self, clusters, X):
        """
        Labeling the samples with index of clusters
        :return: labeled samples
        """
        y_pred = ms.ops.zeros(X.shape[0])
        for idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                y_pred[sample_idx] = idx

        return y_pred

    def fit(self, X):
        """
        Initializing centroid using Kmeans++, then find distance between each sample and initial centroids, then assign
        cluster label based on min_distance, repeat this process for max_iteration and simultaneously updating
        centroid by calculating distance between sample and updated centroid. Convergence happen when difference between
        previous and updated centroid is None.
        :return: updated centroids of the cluster after max_iterations.
        """
        centroids = self.initialize_centroid(X, self.k)
        for _ in range(self.max_iterations):
            clusters = self.create_clusters(centroids, X)
            previous_centroids = centroids
            centroids = self.update_centroids(clusters, X)
            print('centroids ', centroids);
            difference = centroids - previous_centroids

            print('diff ', ms.ops.sum(difference))
            if not difference.numpy().any():
                break

        self.KMeans_Centroids = centroids
        return centroids

    def predict(self, X):
        """
        :return: label/cluster number for each input sample is returned
        """
        if not self.KMeans_Centroids.numpy().any():
            raise Exception("No Centroids Found. Run KMeans fit")

        clusters = self.create_clusters(self.KMeans_Centroids, X)
        labels = self.label_clusters(clusters, X)

        return labels


if __name__ == '__main__':
    iris = load_iris()
    set_seed(0)
    X = iris.data
    y = iris.target
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    x_train = ms.Tensor(x_train).astype(mstype.float32)

    y_train = ms.Tensor(y_train)
    x_test = ms.Tensor(x_test).astype(mstype.float32)
    y_test = ms.Tensor(y_test)
    n_classes = len(ms.ops.unique(y_train))
    print(n_classes)

    tensor = ms.Tensor(np.array([[0, 1], [2, 3]])).astype(mstype.float32)
    print(tensor[0])

    kmeans = KMeans(x_train, k=n_classes, iterations=300)
    kmeans.fit(x_train)
    print('kmeans.KMeans_Centroids.numpy().any()', kmeans.KMeans_Centroids.numpy().any())
    ypred = kmeans.predict(x_test).astype(mstype.int64)
    print(ypred)
    print(y_test.dtype)
    print(f'Accuracy Score: {accuracy_score(y_test.asnumpy(), ypred.asnumpy())}')


