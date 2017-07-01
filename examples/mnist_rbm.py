from tensorflow.contrib.learn.python.learn.datasets.mnist import load_mnist
import numpy as np
from tensorflow_models import rbm
import matplotlib.pyplot as plt

dataset = load_mnist()
# RBM takes binary inputs, so we convert the input to 0/1
binarized_train = (dataset.train.images > 0.5).astype(np.float32)
binarized_test = (dataset.test.images > 0.5).astype(np.float32)

model = rbm.RBM(rbm.RBMLayer.from_shape(784, 500), n_iter=50000)
model.train(binarized_train)
print('Reconstruction error:', model.reconstruction_error(binarized_test))


# Visualize the reconstructions
N_FIGS = 10
rand_idx = np.random.choice(len(binarized_test), N_FIGS, replace=False)
original_images = binarized_test[rand_idx]
reconstructions = model.reconstruct(original_images)
f, axarr = plt.subplots(2, N_FIGS, figsize=(14, 4))
f.suptitle('Upper: original; Lower: RBM reconstruction')
for i in range(N_FIGS):
    axarr[0, i].imshow(original_images[i].reshape(28,28), cmap='gray')
    axarr[0, i].axis('off')
    axarr[1, i].imshow(reconstructions[i].reshape(28,28), cmap='gray')
    axarr[1, i].axis('off')
plt.savefig('mnist_rbm_reconstruction.png', dpi=200)