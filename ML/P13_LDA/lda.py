import mindspore as ms
from mindspore import context, Tensor, nn
from mindspore.ops import operations as ops
from mindspore import dtype as mstype

from NaiveBayes import NaiveBayes
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class LDA:
    def __init__(self):
        self.w = None

    def covariance_matrix(self, X):
        """
        :param X: Input tensor
        :return: cavariance of input tensor
        """
        centering_X = X - ms.ops.mean(X, axis=0)
        cov = ms.ops.mm(centering_X.transpose(), centering_X) / (centering_X.shape[0] - 1)
        return cov

    def fit(self, X, y):
        """
        :param X: Input tensor
        :param y: output tensor
        :return: transformation vector - to convert high dimensional input space into lower dimensional
        subspace.
        X1, X2 are samples based on class. cov_1 and cov_2 measures how features of samples of each class are related.

        """
        #print('y==0: ', (y==0).shape)
        X1 = X[y==0]
        X2 = X[y==1]
        cov_1 = self.covariance_matrix(X1)
        cov_2 = self.covariance_matrix(X2)
        cov_total = cov_1 + cov_2
        #print('cov_total: ', cov_total)
        mean1 = ms.ops.mean(X1, axis=0)
        mean2 = ms.ops.mean(X2, axis=0)
        mean_diff = mean1 - mean2

        # Determine the vector which when X is projected onto it best separates the
        # data by class. w = (mean1 - mean2) / (cov1 + cov2)
        self.w = ms.ops.mm(ms.ops.pinv(cov_total), mean_diff.unsqueeze(1))

    def transform(self, X, y):
        self.fit(X, y)
        X_transformed = ms.ops.mm(X, self.w)
        return X_transformed

    def predict(self, X):
        y_pred = []
        for sample in X:
            h = ms.ops.mm(sample.unsqueeze(0), self.w)
            #print('(h < 0): ', (h < 0))
            y = 1 * (h < 0)
            print('y: ', y)
            y_pred.append(int(y.squeeze().item()))

        return y_pred

if __name__ == '__main__':
    breast_cancer = load_breast_cancer()
    X = breast_cancer.data
    X_normalized = MinMaxScaler().fit_transform(X)
    y = breast_cancer.target
    x_train, x_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2)
    x_train = Tensor(x_train, dtype=mstype.float32)
    x_test = Tensor(x_test, dtype=mstype.float32)
    y_train = Tensor(y_train, dtype=mstype.int32) # .unsqueeze(1)
    y_test = Tensor(y_test, dtype=mstype.int32)
    print('y_train len: ', len(ms.ops.unique(y_train)[0]))

    lda = LDA()
    print(x_train.dtype)
    print(y_train.dtype)
    X_transformed = lda.transform(x_train, y_train)
    print('X_transformed: ', X_transformed.shape)
    GNB = NaiveBayes(X_transformed, y_train)
    GNB.find_mu_and_sigma(X_transformed, y_train)
    X_test_transformed = lda.transform(x_test, y_test)
    y_pred = GNB.predict_probability(X_test_transformed)
    print('y_test: ', y_test)
    print('y_pred: ', y_pred)
    print(f'Accuracy Score: {accuracy_score(y_test.asnumpy(), y_pred.asnumpy())}')

