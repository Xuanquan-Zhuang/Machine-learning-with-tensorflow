import tensorflow as tf

class LogisticRegression(object):
	def __init__(self):
		self.graph = tf.Graph()
		self.sess = tf.Session(graph=self.graph)
		self.input_x = tf.placeholder(tf.float32, shape=[None, None])
		self.input_y = tf.placeholder(tf.float32, shape=[None, ])
		self.weights = None
		self.bias = None
		self.predict = None
		self.loss = None
		self.train = None
	
	def fit(self, x, y):
		n_variables = x.shape[1]
		self.weights = tf.Variable(tf.truncated_normal([n_variables, 1]), dtype=tf.float32)
		self.bias = tf.Variable(tf.constant(0.1), dtype=tf.float32)
		self.predict = tf.add(tf.matmul(self.input_x, self.weights), self.bias)
		self.loss = tf.
