from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tensorflow.contrib.learn.python.learn.datasets.mnist import load_mnist
from tensorflow_models.mlp import MultiLayerPerceptron
from tensorflow_models.logit_reg import LogitReg
from tensorflow_models.dbn import DeepBeliefNet

dataset = load_mnist()

# sklearn logistic regression
sk_logreg = LogisticRegression(C=1, multi_class='multinomial', solver='lbfgs')
sk_logreg.fit(dataset.train.images, dataset.train.labels)
sk_pred = sk_logreg.predict(dataset.test.images)
print('sklearn accuracy', accuracy_score(dataset.test.labels, sk_pred))

# tensorflow logistic regression
tf_logreg = LogitReg(784, n_output_class=10, n_iter=25000)
tf_logreg.train(dataset.train.images, dataset.train.labels)
tf_logreg_pred = tf_logreg.predict(dataset.test.images)
print('TensorFlow LogReg accuracy', accuracy_score(dataset.test.labels, tf_logreg_pred))

# tensorflow MLP
tf_mlp = MultiLayerPerceptron(784, [500, 300], n_output_class=10, n_iter=50000)
tf_mlp.train(dataset.train.images, dataset.train.labels)
tf_mlp_pred = tf_mlp.predict(dataset.test.images)
print('TensorFlow MLP accuracy', accuracy_score(dataset.test.labels, tf_mlp_pred))

# tensorflow DBN
tf_dbn =DeepBeliefNet(784, [500, 300], n_output_class=10, n_iter=50000)
tf_dbn.pretrain(dataset.train.images, n_iter=10000)
tf_dbn.train(dataset.train.images, dataset.train.labels)
tf_dbn_pred = tf_dbn.predict(dataset.test.images)
print('TensorFlow DBN accuracy', accuracy_score(dataset.test.labels, tf_dbn_pred))
