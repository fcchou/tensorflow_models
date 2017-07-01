from tensorflow.contrib.learn.python.learn.datasets.mnist import load_mnist
import numpy as np
from tensorflow_models.autoencoder import Autoencoder
import matplotlib.pyplot as plt

dataset = load_mnist()

model = Autoencoder(784, 1500, n_iter=10000, noise_level=0.2)
model.train(dataset.train.images)
print('Reconstruction error:', model.reconstruction_error(dataset.test.images))

# Visualize the reconstructions
N_FIGS = 10
rand_idx = np.random.choice(len(dataset.test.images), N_FIGS, replace=False)
original_images = dataset.test.images[rand_idx]
reconstructions = model.reconstruct(original_images)
f, axarr = plt.subplots(2, N_FIGS, figsize=(14, 4))
f.suptitle('Upper: original; Lower: reconstruction')
for i in range(N_FIGS):
    axarr[0, i].imshow(original_images[i].reshape(28,28), cmap='gray')
    axarr[0, i].axis('off')
    axarr[1, i].imshow(reconstructions[i].reshape(28,28), cmap='gray')
    axarr[1, i].axis('off')
plt.savefig('mnist_autoencoder_reconstruction.png', dpi=200)