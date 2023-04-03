import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler


class CustomKMeans:
    def __init__(self, k):
        self.k = k
        self.centres = None

    def fit(self, X, eps=1e-6):
        self.centres = X[:self.k]
        current_eps = 1
        while current_eps > eps:
            new_centers = self.calculate_new_centres(X)
            current_eps = np.linalg.norm(np.array(new_centers) - np.array(self.centres))
            self.centres = new_centers

    def predict(self, X):
        return self.find_nearest_centre(X)

    def tune(self, X, p=0.2):
        self.k = 1
        self.fit(X)
        errors = [self.error(X)]
        while True:
            self.k += 1
            self.fit(X)
            error = self.error(X)
            previous_error = errors[-1]
            error_decrease = np.abs(previous_error - error) / previous_error
            if error_decrease < p:
                self.k -= 1
                break
            errors.append(error)

    def find_nearest_centre(self, X):
        nearest_centres = []
        for observation in X:
            distances = np.linalg.norm(observation - self.centres, axis=1)
            nearest_centres.append(np.argmin(distances))
        return nearest_centres

    def calculate_new_centres(self, X):
        means = []
        for i in range(self.k):
            obs_cluster = self.get_observations_in_cluster(i, X)
            means += np.mean(obs_cluster, axis=0).tolist()
        return np.array(means).reshape(self.k, X.shape[1])

    error_decrease = lambda x, y: np.abs(x - y) / x

    def error(self, X):
        error = 0
        for i in range(self.k):
            obs_cluster = self.get_observations_in_cluster(i, X)
            error += np.linalg.norm(obs_cluster - self.centres[i]) ** 2
        return np.sqrt(error / self.k)

    def get_observations_in_cluster(self, number_cluster, X):
        index_centres = self.find_nearest_centre(X)
        index_obs_cluster = [index for index in range(len(index_centres)) if index_centres[index] == number_cluster]
        return X[index_obs_cluster]


def plot_comparison(data: np.ndarray,
                    predicted_clusters: np.ndarray,
                    true_clusters: np.ndarray = None,
                    centers: np.ndarray = None,
                    show: bool = True):
    if true_clusters is not None:
        plt.figure(figsize=(20, 10))

        plt.subplot(1, 2, 1)
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=predicted_clusters, palette='deep')
        if centers is not None:
            sns.scatterplot(x=centers[:, 0], y=centers[:, 1], marker='X', color='k', s=200)
        plt.grid()

        plt.subplot(1, 2, 2)
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=true_clusters, palette='deep')
        if centers is not None:
            sns.scatterplot(x=centers[:, 0], y=centers[:, 1], marker='X', color='k', s=200)
        plt.grid()
    else:
        plt.figure(figsize=(10, 10))
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=predicted_clusters, palette='deep')
        if centers is not None:
            sns.scatterplot(x=centers[:, 0], y=centers[:, 1], marker='X', color='k', s=200)
        plt.grid()

    plt.savefig('Visualization.png', bbox_inches='tight')
    if show:
        plt.show()


if __name__ == '__main__':
    # Load X
    data = load_wine(as_frame=True, return_X_y=True)
    X_full, y_full = data

    # Permutate it to make things more interesting
    rnd = np.random.RandomState(42)
    permutations = rnd.permutation(len(X_full))
    X_full = X_full.iloc[permutations]
    y_full = y_full.iloc[permutations]

    # From dataframe to ndarray
    X_full = X_full.values
    y_full = y_full.values

    # Scale X
    scaler = MinMaxScaler()
    X_full = scaler.fit_transform(X_full)

    # Model
    KMeans = CustomKMeans(k=2)
    KMeans.tune(X_full)
    KMeans.fit(X_full)
    y_pred = KMeans.predict(X_full)
    print(y_pred[:20])
