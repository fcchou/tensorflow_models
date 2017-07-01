import tensorflow as tf

from tensorflow_models.util import MiniBatchData, FCLayer, LogRegLayer


class MultiLayerPerceptron:
    def __init__(
        self,
        input_size,
        hidden_layer_sizes,
        n_output_class=2,
        learning_rate=0.001,
        regularization=0.0001,
        n_iter=5000,
        batch_size=100,
        activation=tf.nn.relu,
    ):
        """Multi-layer perceptron implemented with TensorFlow.

        The training is done with mini-batch gradient descent. L2 regularization is included in the loss.

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
            activation (tensorflow function): Activation function for the layers.
        """
        self.n_iter = n_iter
        self.batch_size = batch_size

        # Input placeholders
        self._x = tf.placeholder(tf.float32, [None, input_size])
        self._labels = tf.placeholder(tf.int64, [None])

        curr_input = self._x
        self._layers = []
        for i, size in enumerate(hidden_layer_sizes):
            curr_layer = FCLayer(curr_input, size, name='hidden{}'.format(i), activation_func=activation)
            curr_input = curr_layer.output
            self._layers.append(curr_layer)

        self._layers.append(LogRegLayer(curr_input, n_output_class))

        l2_loss = sum(l.l2_loss for l in self._layers)

        loss_func = self._layers[-1].cross_entropy(self._labels) + l2_loss * regularization
        self._train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_func)
        self._y = self._layers[-1].labels_pred

        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())

    def train(self, x, labels):
        """Train the model.

        Args:
            x (numpy.ndarray): Feature vectors for the training set.
            labels (numpy.nadarray): Labels of the training set.
        """
        data = MiniBatchData([x, labels], batch_size=self.batch_size)
        for _ in range(self.n_iter):
            batch_xs, batch_ys = data.next()
            self._sess.run(self._train_step, feed_dict={self._x: batch_xs, self._labels: batch_ys})

    def predict(self, x):
        """Predict the labels with trained model.

        Args:
            x(numpy.ndarray): Input feature vectors.

        Returns:
            1D numpy array for the predicted labels.
        """
        return self._sess.run(self._y, feed_dict={self._x: x})