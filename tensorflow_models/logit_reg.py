import tensorflow as tf

from tensorflow_models.util import MiniBatchData, LogRegLayer


class LogitReg:
    def __init__(
        self,
        input_size,
        n_output_class=2,
        learning_rate=0.001,
        regularization=0.0001,
        n_iter=5000,
        batch_size=100,
    ):
        """Logistic regression model implemented with TensorFlow.

        The training is done with mini-batch gradient descent. L2 regularization is included in the loss.

        Args:
            input_size (int): Number of features in the input
            n_output_class (int):
                Number of output classes in the labels. The training labels must be in [0, n_output_class).
            learning_rate (float): Learning rate of the gradient descent.
            regularization (float): Strength of the L2 regularization. Use 0 to skip regularization.
            n_iter (int): Positive integer. The number of gradient descent iterations.
            batch_size (int): Size of the mini batch.
        """
        self.n_iter = n_iter
        self.batch_size = batch_size

        # Input placeholders
        self._x = tf.placeholder(tf.float32, [None, input_size])
        self._labels = tf.placeholder(tf.int64, [None])

        self._layer = LogRegLayer(self._x, n_output_class)
        loss_func = self._layer.cross_entropy(self._labels) + self._layer.l2_loss * regularization
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_func)
        self._y = self._layer.labels_pred
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
            self._sess.run(self.train_step, feed_dict={self._x: batch_xs, self._labels: batch_ys})

    def predict(self, x):
        """Predict the labels with trained model.

        Args:
            x(numpy.ndarray): Input feature vectors.

        Returns:
            1D numpy array for the predicted labels.
        """
        return self._sess.run(self._y, feed_dict={self._x: x})

