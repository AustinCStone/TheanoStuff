import theano
from load import mnist
from theano import tensor as T
import numpy as np
from theano import function
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

srng = RandomStreams()

def dropout(X, drop_prob=0.):
	if drop_prob > 0:
		retain_prob = 1 - drop_prob
		X *= srng.binomial(X.shape, p=retain_prob)
		X /= retain_prob
	return X

def build_weights(layer_sizes, initial_weight_max=0.01):
	weights = []
	for layer_size in layer_sizes:
		weights.append(theano.shared(np.random.randn(*layer_size) * initial_weight_max))
	return weights

# currently not weighing in previous gradients
def RMSprop(cost, params, lr=0.001, epsilon=1e-6):
	grads = T.grad(cost=cost, wrt=params)
	updates = []
	for (grad, param) in zip(grads, params):
		grad_scaling = T.sqrt(grad ** 2 + epsilon)
		updates.append((param, param - lr * (grad / grad_scaling)))
	return updates

def forward_prop(layers, drop_probs, p_drop_input=0.2, p_drop_hidden=0.5):
	dropout_outputs = []
	prev_layer = None
	prev_drop_prob = None
	current_layer = 0
	for (layer, drop_prob) in zip(layers, drop_probs):
		if current_layer == 0: # input layer
			prev_layer = layer
			prev_drop_prob = drop_prob
			current_layer += 1
			continue
		prev_layer = dropout(prev_layer, prev_drop_prob)
		layer_output = None
		if current_layer == len(layers) - 1: # output layer
			layer_output = T.nnet.softmax(T.dot(prev_layer, layer))
		else:	
			layer_output = T.maximum((T.dot(prev_layer, layer)), 0) # not sure why relu isn't built in
		dropout_outputs.append(layer_output)
		prev_layer = layer_output
		prev_drop_prob = drop_prob
		current_layer += 1
	return dropout_outputs


def run_net(trials, batch_size):
	train_x, test_x, train_y, test_y = mnist(onehot=True)
	input_dim = len(train_x[0])
	output_dim = len(test_y[0])
	[w_h1, w_h2, weight_outputs] = build_weights([(input_dim, 625), (625, 625), (625, output_dim)])
	X = T.fmatrix() #symbolic variable for weight matrix
	Y = T.fmatrix() #symbolic variable for output
	# outputs from the layers with dropout
	[dropout_h1, dropout_h2, dropout_net_output] = forward_prop([X, w_h1, w_h2, weight_outputs], [0.2, 0.5, 0.5, 0.5])
	# outputs from the layers without dropout
	[h1, h2, net_output] = forward_prop([X, w_h1, w_h2, weight_outputs], [0., 0., 0., 0.])
	# actual prediction
	predicted_label = T.argmax(net_output, axis=1)
	cost = T.mean(T.nnet.categorical_crossentropy(dropout_net_output, Y))
	updates = RMSprop(cost, [w_h1, w_h2, weight_outputs])
	net_train = function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
	get_net_output = function(inputs=[X], outputs=predicted_label, allow_input_downcast=True)

	for trial in range(trials):
		for batch_start in range(0, len(train_x) - batch_size, batch_size):
			batch_end = batch_start + batch_size
			net_train(train_x[batch_start:batch_end], train_y[batch_start:batch_end])
		print np.mean(np.argmax(test_y, axis=1) == get_net_output(test_x))

if __name__ == "__main__":
    run_net(100, 128)
