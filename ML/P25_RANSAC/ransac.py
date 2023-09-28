import mindspore as ms
from mindspore import context, Tensor, nn

import math
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

ms.set_seed(42)
context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

class LeastSquareModel:
    def fit(self, A, Y):
        A_T = A.transpose()
        A_T_A = ms.ops.mm(A_T, A)
        A_T_Y = ms.ops.mm(A_T, Y)
        model = ms.ops.mm(ms.ops.pinv(A_T_A), A_T_Y)

        return model

class RansacModel:
    def __init__(self, curve_fitting_model):
        self.curve_fitting_model = curve_fitting_model

    def fit(self, A, Y, num_sample, threshold):
        num_iterations = math.inf
        iterations_done = 0
        num_samples = 3
        max_inlier_count = 0
        best_model = None
        probability_outlier = ms.ops.scalar_to_tensor(0.5, dtype=ms.float32)
        desired_prob = ms.ops.scalar_to_tensor(0.95, dtype=ms.float32)
        total_data = ms.ops.column_stack((A, Y))
        #data_size = len(total_data)

        while num_iterations > iterations_done:
            total_data = ms.ops.shuffle(total_data)

            sample_data = total_data[:num_samples, :]
            estimated_model = self.curve_fitting_model.fit(sample_data[:, :-1], sample_data[:, -1:])
            y_cap = ms.ops.mm(A, estimated_model)
            error = ms.ops.abs(Y - y_cap.transpose())
            data_size = error.shape[0]*error.shape[1]
            t = Tensor(error < threshold).int()
            inlier_count = Tensor(ms.ops.count_nonzero(t))

            if int(inlier_count.item()) > max_inlier_count:
                max_inlier_count = int(inlier_count.item())
                best_model = estimated_model

            probability_outlier = 1 - int(inlier_count.item())*1.0/data_size
            num_iterations = ms.ops.log(Tensor(1 - desired_prob)) / ms.ops.log(
                Tensor(1 - (1 - probability_outlier) ** num_sample))
            iterations_done = iterations_done + 1

            #print('# s:', iterations_done)
            #print('# n:', num_iterations)
            #print('# max_inlier_count: ', max_inlier_count)

        return best_model

def fit_curve(X, y):
    x_square = ms.ops.pow(X, 2)

    A = ms.ops.stack((x_square, X, ms.ops.ones(X.shape[0]).unsqueeze(1)), axis=1)
    A = A.squeeze(2)
    threshold = ms.ops.std(y) / 5
    ls_model = LeastSquareModel()
    ls_model_estimate = ls_model.fit(A, y)

    ls_model_y = ms.ops.mm(A, ls_model_estimate)

    ransac_model = RansacModel(ls_model)
    ransac_model_estimate = ransac_model.fit(A, y, 3, threshold)
    ransac_model_y = ms.ops.mm(A, ransac_model_estimate)

    return ls_model_y, ransac_model_y

if __name__ == '__main__':

    X1, y1 = make_regression(n_features=1, n_targets=1)
    X2, y2 = make_regression(n_features=1, n_targets=1)

    # X1, y1 = data1['x '], data1['y']
    # X2, y2 = data2['X'], data2['y']
    X1, y1 = Tensor(X1, dtype=ms.float32), Tensor(y1, dtype=ms.float32).unsqueeze(1)
    X2, y2 = Tensor(X2, dtype=ms.float32), Tensor(y2, dtype=ms.float32).unsqueeze(1)
    ls_model_y1, ransac_model_y1 = fit_curve(X1, y1)
    ls_model_y2, ransac_model_y2 = fit_curve(X2, y2)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.set_title('Dataset-1')
    ax1.scatter(X1, y1, marker='o', color=(0, 1, 0), label='data points')
    ax1.plot(X1, ls_model_y1, color='red', label='Least sqaure model')
    ax1.plot(X1, ransac_model_y1, color='blue', label='Ransac model')
    ax1.set(xlabel='x-axis', ylabel='y-axis')
    ax1.legend()

    ax2.set_title('Dataset-2')
    ax2.scatter(X2, y2, marker='o', color=(0, 1, 0), label='data points')
    ax2.plot(X2, ls_model_y2, color='red', label='Least sqaure model')
    ax2.plot(X2, ransac_model_y2, color='blue', label='Ransac model')
    ax2.set(xlabel='x-axis', ylabel='y-axis')
    ax2.legend()

    plt.show()
    exit(0)
