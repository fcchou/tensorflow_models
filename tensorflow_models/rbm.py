import tensorflow as tf
import numpy as np
from tensorflow_models.util import MiniBatchData


class RBM:
    def __init__(
        self,
        rbm_layer,
        learning_rate=0.001,
        n_iter=5000,
        batch_size=100,
        negative_sample_size=100,
        regularization=0.00001,
        cd_k=1,
        session=None,
    ):
        """Restricted Boltzmann Machine by Tensorflow.
        Both the visible and hidden layers are binary units.
        The training is performed with the standard contrastive divergence (CD). In additional the persistent-CD
        algorithm is used.

        Args:
            rbm_layer (RBMLayer): RBMLayer that defines the RBM structure
            learning_rate (float): Learning rate of the gradient descent.
            n_iter (int): Number of gradient descent iterations.
            batch_size (int): Size of the mini batch.
            negative_sample_size (int): Number of negative sample particles to kept during CD for each iteration.
            regularization (float): Strength of the L2 regularization. Use 0 to skip regularization.
            cd_k (int): Number of CD steps to perform.
            session (tf.Session): A Tensorflow session. If not given, creates a new session and initialize.
        """
        curr_vars = set(tf.global_variables())

        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.batch_size = batch_size
        self._regularization = regularization

        self._rbm_layer = rbm_layer
        self._x = tf.placeholder(tf.float32, [None, self._rbm_layer.visible_size])  # Input units
        # Negative particles placeholder
        self._negative_v = tf.placeholder(tf.float32, [negative_sample_size, self._rbm_layer.visible_size])

        # Training and negative particle sampler
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self._cost_func())
        self._persisted_v_updater = self._rbm_layer.gibbs_sample(self._negative_v, cd_k)
        self._persisted_v = None

        if session is None:
            self._sess = tf.Session()
            self._sess.run(tf.global_variables_initializer())
        else:
            self._sess = session
            extra_vars = set(tf.global_variables()) - curr_vars
            self._sess.run(tf.variables_initializer(extra_vars))

    def _cost_func(self):
        cost = (
            tf.reduce_mean(self._rbm_layer.free_energy(self._x)) -
            tf.reduce_mean(self._rbm_layer.free_energy(self._negative_v)) +
            tf.nn.l2_loss(self._rbm_layer.weight) * self._regularization
        )
        return cost

    def reconstruction_error(self, input_x):
        """Compute the model reconstruction error as a proxy of the model accuracy.

        Args:
            input_x (array-like): Input data to be used for error estimation (e.g. validation set).

        Returns:
            The log-loss reconstruction error.
        """
        return self._sess.run(self._rbm_layer.reconstruction_error(input_x))

    def train(self, x):
        """Train the model.

        Args:
            x (numpy.ndarray): Feature vectors for the training set.
        """
        if self._persisted_v is None:
            # The negative particles (self._persisted_v) is initialized by running 20 steps of CD
            persisted_v = self._rbm_layer.gibbs_sample(
                tf.constant(
                    np.random.randint(0, 1, size=self._negative_v.get_shape().as_list()),
                    dtype=tf.float32,
                ),
                n_step=20,
            )
            self._persisted_v = self._sess.run(persisted_v)

        data = MiniBatchData([x], batch_size=self.batch_size)
        for _ in range(self.n_iter):
            batch_xs, = data.next()
            self._persisted_v = self._sess.run(
                self._persisted_v_updater,
                feed_dict={self._negative_v: self._persisted_v},
            )
            self._sess.run(self.train_step, feed_dict={self._x: batch_xs, self._negative_v: self._persisted_v})

    def reconstruct(self, input_x, n_step=1000):
        """Reconstruct the input vector by multi-step Gibbs sampling.

        Args:
            input_x (array-like): Input data to be reconstructed.
            n_step (int): Number of Gibbs sampling steps

        Returns:
            Reconstructed vectors, same shape as input_x.
        """
        return self._sess.run(self._rbm_layer.gibbs_sample(tf.constant(input_x, dtype=tf.float32), n_step))


class RBMLayer:
    def __init__(
        self,
        weight,
        hbias,
        vbias,
        name='rbm-layer',
    ):
        """Restricted Boltzmann Machine Layer.

        Both the visible and hidden layers are binary units.

        Args:
            weight (tf.Variable): Weights of RBM.
            hbias (tf.Variable): Bias on hidden units.
            vbias (tf.Variable): Bias on visible unit.
            name (str): Name of the RBM layer.

        Raises:
            ValueError if the weight is not of shape [visible_size, hidden_size]
        """
        self.visible_size = vbias.get_shape().as_list()[0]
        self.hidden_size = hbias.get_shape().as_list()[0]

        if weight.get_shape().as_list() != [self.visible_size, self.hidden_size]:
            raise ValueError('Incompatible weight and hbias/vias')

        # Model parameters
        with tf.name_scope(name):
            self.vbias = vbias
            self.hbias = hbias
            self.weight = weight

    @classmethod
    def from_shape(cls, visible_size, hidden_size, name='rbm-layer'):
        """Construct an RBMLayer with visible_size and hidden_size.

        Both the visible and hidden layers are binary units.

        Args:
            visible_size (int): Size of the visible layer.
            hidden_size (int): Size of the hidden layer.
            name (str): Name of the RBM layer.
        """
        vbias = tf.Variable(tf.zeros([visible_size]))
        hbias = tf.Variable(tf.zeros([hidden_size]))
        weight_init_std = 4.0 * np.sqrt(6.0 / (visible_size + hidden_size))
        weight = tf.Variable(tf.truncated_normal([visible_size, hidden_size], stddev=weight_init_std))
        return cls(weight, hbias, vbias, name=name)

    def _visible2hidden(self, v):
        prob = tf.nn.sigmoid(tf.matmul(v, self.weight) + self.hbias)
        return self._sample_prob(prob), prob

    def _hidden2visible(self, h):
        prob = tf.nn.sigmoid(tf.matmul(h, tf.transpose(self.weight)) + self.vbias)
        return self._sample_prob(prob), prob

    def _reconstruct_v(self, v):
        h, _ = self._visible2hidden(v)
        v_new, prob = self._hidden2visible(h)
        return v_new, prob

    def free_energy(self, input_v):
        """compute free energy of a input visible vector given RBM parameters.

        Args:
            input_v (array-like): Input visible vector

        Returns (tf.Tensor): Free energy.
        """
        return -(
            tf.reduce_sum(input_v * self.vbias, 1) +
            tf.reduce_sum(tf.log(1 + tf.exp(tf.matmul(input_v, self.weight) + self.hbias)), 1)
        )

    def reconstruction_error(self, input_v):
        """Compute the reconstruction error as a proxy of the model accuracy.

        Args:
            input_v (tf.Tensor): Input data to be used for error estimation (e.g. validation set).

        Returns (tf.Tensor):
            The log-loss reconstruction error.
        """
        _, p = self._reconstruct_v(input_v)

        # The log-loss error diverges when p == 0 or p == 1. Apply a min/max for the probability
        p = tf.clip_by_value(p, 1e-5, 1 - 1e-5)
        return -tf.reduce_mean(input_v * tf.log(p) + (1 - input_v) * tf.log(1 - p))

    def gibbs_sample(self, input_v, n_step=1000):
        """Reconstruct the input vector by multi-step Gibbs sampling.

        Args:
            input_v (tf.Tensor): Input visible vector to be reconstructed.
            n_step (int): Number of Gibbs sampling steps

        Returns (tf.Tensor):
            Reconstructed vectors, same shape as input_x.
        """
        v = input_v
        for i in range(n_step):
            v, _ = self._reconstruct_v(v)
        return v

    @staticmethod
    def _sample_prob(probs):
        """Generate binary samples given probability"""
        return tf.to_float(probs > tf.random_uniform(probs.get_shape()))