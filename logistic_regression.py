import theano
from load import mnist
from theano import tensor as T
import numpy as np

INITIAL_WEIGHT_MAX = .01
LEARNING_RATE = 0.01

def run_regression(trials, batch_size):
	train_x, test_x, train_y, test_y = mnist(onehot=True)
	x_dim = len(train_x[0])
	y_dim = len(train_y[0])
	weight_vec = theano.shared(np.random.randn(x_dim, y_dim) * INITIAL_WEIGHT_MAX)
	offset = theano.shared(np.zeros(y_dim))
	input_data = T.fmatrix('input_data')
	label = T.fmatrix('label')
	softmax_output = T.nnet.softmax(T.dot(input_data, weight_vec) + offset)
	cost = T.mean(T.nnet.categorical_crossentropy(softmax_output, label))
	weight_grad, offset_grad = T.grad(cost=cost, wrt=[weight_vec, offset])
	updates = [[weight_vec, weight_vec - weight_grad * LEARNING_RATE], [offset, offset - offset_grad * LEARNING_RATE]]
	train_f = theano.function(inputs=[input_data, label], outputs=cost, updates=updates, allow_input_downcast=True)
	predicted_label = T.argmax(softmax_output, axis=1)
	output_f = theano.function(inputs=[input_data], outputs=predicted_label, allow_input_downcast=True)

	for i in range(trials):
		for start, end in zip(range(0, len(train_x), batch_size), range(batch_size, len(train_x), batch_size)):
			cost = train_f(train_x[start:end], train_y[start:end])
		print i, np.mean(np.argmax(test_y, axis=1) == output_f(test_x))



