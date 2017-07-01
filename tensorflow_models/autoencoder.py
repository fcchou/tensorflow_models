import tensorflow as tf
from tensorflow_models.util import MiniBatchData, FCLayer


class Autoencoder:
    def __init__(
        self,
        input_size,
        n_hidden_unit,
        learning_rate=0.001,
        n_iter=5000,
        batch_size=100,
        noise_level=0,
    ):
        """Denoising autoencoder. The input must be in [0, 1].

        Args:
            input_size (int): Number of features in the input
            n_hidden_unit (int): Size of the hidden layers.
            learning_rate (float): Learning rate of the gradient descent.
            n_iter (int): Number of gradient descent iterations.
            batch_size (int): Size of the mini batch.
            noise_level (float): Must be in [0, 1]. The level of noise to be added. 0 stands for no noise,
                1 is complete nullification of the input.
        """
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.noise_level = noise_level

        self._x = tf.placeholder(tf.float32, [None, input_size])
        self._keep_prob = tf.placeholder(tf.float32)
        x_dropout = tf.nn.dropout(self._x, self._keep_prob)
        encode_layer = FCLayer(x_dropout, n_hidden_unit, name='encode_layer', activation_func=tf.nn.relu)
        self._hidden_value = encode_layer.output
        decode_layer = FCLayer(encode_layer.output, input_size, name='decode_layer', activation_func=tf.nn.sigmoid)
        self._reconstruction = decode_layer.output

        # Use reconstruction cross entropy as cost function; this is sensible since input is in [0, 1]
        self._cost_func = -tf.reduce_mean(
            tf.reduce_sum(
                self._x * tf.log(self._reconstruction) + (1 - self._x) * tf.log(1 - self._reconstruction),
                axis=[1],
            )
        )

        self._train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self._cost_func)
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())

    def train(self, x):
        """Train the model.

        Args:
            x (numpy.ndarray): Feature vectors for the training set.
        """
        data = MiniBatchData([x], batch_size=self.batch_size)
        for _ in range(self.n_iter):
            batch_xs, = data.next()
            self._sess.run(self._train_step, feed_dict={self._x: batch_xs, self._keep_prob: (1 - self.noise_level)})

    def reconstruct(self, input_x):
        """Reconstruct the input vector.

        Args:
            input_x (array-like): Input data to be encoded.

        Returns: Reconstructed vectors, same shape as input_x.
        """
        return self._sess.run(self._reconstruction, feed_dict={self._x: input_x, self._keep_prob: 1})

    def encode(self, input_x):
        """Encode the input vector.

        Args:
            input_x (array-like): Input data to be encoded.

        Returns: Encoded input (the values of the hidden layer).
        """
        return self._sess.run(self._hidden_value, feed_dict={self._x: input_x, self._keep_prob: 1})

    def reconstruction_error(self, input_x):
        """Compute the reconstruction error on the input vector.

        Args:
            input_x (array-like): Input data to be encoded.

        Returns: The reconstruction error (float scalar).
        """
        return self._sess.run(self._cost_func, feed_dict={self._x: input_x, self._keep_prob: 1})