import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

NUM_SAMPLES = 100

data = open('training_count_feat_mat_and_vectorizer.pickle', 'rb')
data = pickle.load(data)

sparse_mat = data[0]
features = sparse_mat.toarray()
labels = data[1]
half_samples = NUM_SAMPLES // 2
restricted_features = np.concatenate((features[:half_samples], features[-half_samples:]))
count_vectorizer = data[2]

# normalizing the input features
features = StandardScaler().fit_transform(restricted_features)

pca = PCA()

components = pca.fit(restricted_features)

variances = components.explained_variance_ratio_
plt.plot(variances)
plt.tick_params
plt.show()

print('done')