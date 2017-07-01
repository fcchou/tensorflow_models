import numpy as np
import tensorflow as tf


class MiniBatchData:
    def __init__(self, data_list, batch_size=100, shuffle=True):
        """Create mini batches from the input data

        Args:
            data_list (list):
                List of data (numpy arrays or list) to be batched. The length of each element should be the same.
            batch_size (int):
                Size of the mini batch to be output. Must be a positive integer.
            shuffle (bool):
                Whether to perform random shuffling of the data.
        """
        if len(set(len(d) for d in data_list)) != 1:
            raise ValueError('Sizes of the input data are different')
        if batch_size < 1:
            raise ValueError('Batch size must a positive int')

        self.batch_size = batch_size
        self._idx = 0
        self._len = len(data_list[0])

        if shuffle:
            idx = np.arange(self._len)
            np.random.shuffle(idx)
            self._data_list = [d[idx] for d in data_list]
        else:
            self._data_list = data_list

    def next(self):
        """Get the next mini batch.

        Returns (list):
            List of mini-batched data. The length of the list is the same as the input data_list.
            The size of each element equals to the batch_size.
        """
        start_idx = self._idx
        end_idx = self._idx + self.batch_size
        self._idx = end_idx % self._len
        if end_idx <= self._len:
            return [d[start_idx:end_idx] for d in self._data_list]
        else:
            all_idx = np.arange(start_idx, end_idx) % self._len
            return [d[all_idx] for d in self._data_list]


class FCLayer:
    def __init__(
        self,
        input_tensor,
        output_size,
        name='fc-layer',
        weight_init_std=None,
        activation_func=None,
    ):
        """A fully connected network layer.

        Args:
            input_tensor (tf.Tensor): The input tensor to the layer.
            output_size (int): Size of the layer's output.
            name (str): Name of the layer.
            weight_init_std (float):
                The stddev for the initial weight of the layer (randomly initialized with normal distribution).
                If None it uses a simple heuristic of sqrt(6.0 / (input_size + output_size)).
            activation_func (function):
                Activation function applied to the output. If None, no activation is applied (linear layer).
        """
        self.input = input_tensor
        self.input_size = input_tensor.get_shape().as_list()[1]
        with tf.name_scope(name):
            if weight_init_std is None:
                weight_init_std = np.sqrt(2.0 / self.input_size)
            self.weight = tf.Variable(tf.truncated_normal([self.input_size, output_size], stddev=weight_init_std))
            self.bias = tf.Variable(tf.zeros([output_size]))
            lin_output = tf.matmul(input_tensor, self.weight) + self.bias
            if activation_func is None:
                self.output = lin_output
            else:
                self.output = activation_func(lin_output)

    @property
    def l2_loss(self):
        """Return the l2 loss of the current layer's weight"""
        return tf.nn.l2_loss(self.weight)


class LogRegLayer(FCLayer):
    def __init__(
        self,
        input_tensor,
        output_size=2,
        name='logreg-layer',
    ):
        """A logistic regression layer.

        Args:
            input_tensor (tf.Tensor): The input tensor to the layer.
            output_size (int): Size of the layer's output (i.e. the number of output classes).
            name (str): Name of the layer.
        """
        super().__init__(
            input_tensor,
            output_size=output_size,
            name=name,
            weight_init_std=0,
            activation_func=None,
        )

    @property
    def softmax(self):
        """Softmax of the layer output."""
        return tf.nn.softmax(self.output)

    @property
    def labels_pred(self):
        """Most probable label predicted."""
        return tf.argmax(self.softmax, 1)

    def cross_entropy(self, true_labels):
        """Cross entropy of the predictions to the true labels

        Args:
            true_labels (tf.Tensor): True labels of the data, an 1D Tensor for the integer labels (e.g. [1, 0, 1, 2])

        Returns (tf.Tensor):
            Cross entropy
        """
        return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output, labels=true_labels)