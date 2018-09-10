# coding=utf-8
import tensorflow as tf
slim = tf.contrib.slim

def npn_net(inputs,
            num_classes=1000,
            dropout_keep_prob = 0.8,
            num_net = 0,
            scope='Npn'):
	with tf.variable_scope(scope):
		net = []
		index = 0
		print inputs
		for input_one in inputs:
			if index >= num_net:
				break
			net_tmp = slim.conv2d(input_one, 2048, [1, 1],scope='Conv2d_1a_1x1_'+str(index))
			#net_tmp = slim.dropout(net_tmp, keep_prob=dropout_keep_prob, scope='Dropout_end'+str(index))
			net.append(net_tmp)
			index += 1

		net = tf.concat(net,1)
		print 'net before conv_end\n', net
		net = slim.conv2d(net, 2048, [num_net, 1],padding = 'VALID', scope='Conv2d_end')
		#net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_end')
		print 'net end net\n', net

		logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=tf.nn.relu,
	                             normalizer_fn=slim.batch_norm, scope='logits')
		logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze_logits')
		print 'return\n',logits

	return logits