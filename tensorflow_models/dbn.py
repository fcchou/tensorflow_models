import tensorflow as tf

from tensorflow_models.mlp import MultiLayerPerceptron
from tensorflow_models.rbm import RBMLayer, RBM


class DeepBeliefNet(MultiLayerPerceptron):
    def __init__(
        self,
        input_size,
        hidden_layer_sizes,
        n_output_class=2,
        learning_rate=0.001,
        regularization=0.0001,
        n_iter=50000,
        batch_size=100,
    ):
        """Deep belief network implemented with TensorFlow.

        It works like a multi-layer perceptron, but the layers are first pretrained by RBM.
        The input data is assumed to be in [0, 1]

        Args:
            input_size (int): Number of features in the input
            hidden_layer_sizes (list):
                List of int representing the sizes of the hidden layers. The list length determines the depth of the
                network.
            n_output_class (int):
                Number of output classes in the labels. The training labels must be in [0, n_output_class).
            learning_rate (float): Learning rate of the gradient descent.
            regularization (float): Strength of the L2 regularization. Use 0 to skip regularization.
            n_iter (int): Positive integer. The number of gradient descent iterations.
            batch_size (int): Size of the mini batch.
        """
        super().__init__(
            input_size=input_size,
            hidden_layer_sizes=hidden_layer_sizes,
            n_output_class=n_output_class,
            learning_rate=learning_rate,
            regularization=regularization,
            n_iter=n_iter,
            batch_size=batch_size,
            activation=tf.nn.sigmoid,
        )

        # Skip the last log-reg layer for RBM pretraining
        self._rbm_layers = [
            RBMLayer(
                layer.weight,
                layer.bias,
                tf.Variable(tf.zeros([layer.input_size])),
            )
            for layer in self._layers[:-1]
        ]
        self._sess.run(tf.global_variables_initializer())

    def pretrain(
        self,
        x,
        learning_rate=0.001,
        n_iter=5000,
        batch_size=100,
        negative_sample_size=100,
        regularization=0.00001,
        cd_k=1,
    ):
        """Layer-wise RBM pretraining.

        Args:
            x (numpy.ndarray): Feature vectors for the training set.
            learning_rate (float): Learning rate of the gradient descent.
            n_iter (int): Number of gradient descent iterations.
            batch_size (int): Size of the mini batch.
            negative_sample_size (int): Number of negative sample particles to kept during CD for each iteration.
            regularization (float): Strength of the L2 regularization. Use 0 to skip regularization.
            cd_k (int): Number of CD steps to perform.
        """
        training_x = x
        for layer_idx, rbm_layer in enumerate(self._rbm_layers):
            # Pretrain the layer
            rbm_training = RBM(
                rbm_layer,
                learning_rate=learning_rate,
                n_iter=n_iter,
                batch_size=batch_size,
                negative_sample_size=negative_sample_size,
                regularization=regularization,
                cd_k=cd_k,
                session=self._sess,
            )
            rbm_training.train(training_x)

            # Get the input for the next_layer
            training_x = self._sess.run(
                self._layers[layer_idx].output,
                feed_dict={self._x: x},
            )