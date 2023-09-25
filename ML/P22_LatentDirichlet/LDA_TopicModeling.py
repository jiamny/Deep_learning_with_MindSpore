import mindspore as ms
from mindspore import context, Tensor, nn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

ms.set_seed(42)
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

class LatentDirichlet:
    def __init__(self, D, V, T):
        self.D = len(D)
        self.V = len(V)
        self.T = T
        self.alpha = 1 / T
        self.beta = 1 / T

    def fit_transform(self, documents):
        z_d_n = [[0 for _ in range(len(d))] for d in documents]
        theta_d_z = ms.ops.zeros((self.D, self.T))
        phi_z_w = ms.ops.zeros((self.T, self.V))
        n_z = ms.ops.zeros((self.T))
        n_d = ms.ops.zeros((self.D))

        for d, doc in enumerate(documents):
            for n, w in enumerate(doc):
                z_d_n[d][n] = n % self.T
                z = z_d_n[d][n]
                theta_d_z[d][z] += 1
                phi_z_w[z, w] += 1
                n_z[z] += 1
                n_d[d] += 1
            if (d+1) % 50 == 0:
                print( 'doc: ', d )

        for iter in range(10):
            for d, doc in enumerate(documents):
                for n,w in enumerate(doc):
                    z = z_d_n[d][n]
                    theta_d_z[d][z] -= 1
                    phi_z_w[z, w] -= 1
                    n_z[z] -= 1
                    p_d_t = (theta_d_z[d] + self.alpha) / (n_d[d] - 1 + self.T * self.alpha)
                    p_t_w = (phi_z_w[:, w] + self.beta) / (n_z + self.V * self.beta)
                    p_z = p_d_t * p_t_w
                    p_z /= ms.ops.sum(p_z)
                    new_z = ms.ops.multinomial(p_z, 1)
                    z_d_n[d][n] = new_z[0]
                    theta_d_z[d][new_z] += 1
                    phi_z_w[new_z, w] += 1
                    n_z[new_z] += 1
                    if (d+1) % 50 == 0:
                        print('Iter: ', (iter+1), ' doc: ', d)

        return theta_d_z, phi_z_w

if __name__ == '__main__':
    n_samples = 10000
    documents = []
    data, _ = fetch_20newsgroups(shuffle=True, random_state=2,
                                 remove=('headers', 'footers', 'quotes'), return_X_y=True)
    print('data: ', len(data))
    data_samples = data[:n_samples]
    print('data_samples: ', data_samples[0:10])
    cnt_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                 max_features=10000,
                                 stop_words='english')

    vectorizer = cnt_vectorizer.fit_transform(data_samples)
    vocabulary = cnt_vectorizer.vocabulary_
    for row in vectorizer.toarray():
        present_words = np.where(row != 0)[0].tolist()
        present_words_with_count = []
        for w_i in present_words:
            for count in range(row[w_i]):
                present_words_with_count.append(w_i)
        documents.append(present_words_with_count)

    print('documents: ', len(documents))
    print('vocabulary: ', type(vocabulary))

    LD = LatentDirichlet(D=documents, V=vocabulary, T=20)
    topic_distribution, word_distribution = LD.fit_transform(documents)
    i = 1
    plt.plot(topic_distribution[i] / sum(topic_distribution[i]))
    plt.title("Topic distribution $theta_i$ for document {}".format(i))
    plt.show()
    exit(0)