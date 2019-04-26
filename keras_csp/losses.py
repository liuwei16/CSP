from keras import backend as K
from keras.objectives import categorical_crossentropy

if K.image_dim_ordering() == 'tf':
	import tensorflow as tf

epsilon = 1e-4


def cls_center(y_true, y_pred):

	classification_loss = K.binary_crossentropy(y_pred[:, :, :, 0], y_true[:, :, :, 2])
	# firstly we compute the focal weight
	positives = y_true[:, :, :, 2]
	negatives = y_true[:, :, :, 1]-y_true[:, :, :, 2]
	foreground_weight = positives * (1.0 - y_pred[:, :, :, 0]) ** 2.0
	# foreground_weight = positives
	background_weight = negatives * ((1.0 - y_true[:, :, :, 0])**4.0)*(y_pred[:, :, :, 0] ** 2.0)
	# background_weight = negatives * ((1.0 - y_true[:, :, :, 0])**4.0)*(0.01 ** 2.0)

	# foreground_weight = y_true[:, :, :, 0] * (1- y_pred[:, :, :, 0]) ** 2.0
	# background_weight = negatives * y_pred[:, :, :, 0] ** 2.0

	focal_weight = foreground_weight + background_weight

	assigned_boxes = tf.reduce_sum(y_true[:, :, :, 2])
	class_loss = 0.01*tf.reduce_sum(focal_weight*classification_loss) / tf.maximum(1.0, assigned_boxes)

	# assigned_boxes = tf.reduce_sum(tf.reduce_sum(y_true[:, :, :, 1], axis=-1), axis=-1)
	# class_loss = tf.reduce_sum(tf.reduce_sum(classification_loss, axis=-1), axis=-1) / tf.maximum(1.0, assigned_boxes)

	return class_loss

def regr_h(y_true, y_pred):

	absolute_loss = tf.abs(y_true[:, :, :, 0] - y_pred[:, :, :, 0])/(y_true[:, :, :, 0]+1e-10)
	square_loss = 0.5 * ((y_true[:, :, :, 0] - y_pred[:, :, :, 0])/(y_true[:, :, :, 0]+1e-10)) ** 2

	l1_loss = y_true[:, :, :, 1]*tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)

	assigned_boxes = tf.reduce_sum(y_true[:, :, :, 1])
	class_loss = tf.reduce_sum(l1_loss) / tf.maximum(1.0, assigned_boxes)

	return class_loss

def regr_hw(y_true, y_pred):
	absolute_loss = tf.abs(y_true[:, :, :, :2] - y_pred[:, :, :, :]) / (y_true[:, :, :, :2] + 1e-10)
	square_loss = 0.5 * ((y_true[:, :, :, :2] - y_pred[:, :, :, :]) / (y_true[:, :, :, :2] + 1e-10)) ** 2
	loss = y_true[:, :, :, 2] * tf.reduce_sum(tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5),axis=-1)
	assigned_boxes = tf.reduce_sum(y_true[:, :, :, 2])
	class_loss = tf.reduce_sum(loss) / tf.maximum(1.0, assigned_boxes)

	return class_loss

def regr_offset(y_true, y_pred):

	absolute_loss = tf.abs(y_true[:, :, :, :2] - y_pred[:, :, :, :])
	square_loss = 0.5 * (y_true[:, :, :, :2] - y_pred[:, :, :, :]) ** 2
	l1_loss = y_true[:, :, :, 2] * tf.reduce_sum(tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5), axis=-1)

	assigned_boxes = tf.reduce_sum(y_true[:, :, :, 2])
	class_loss = 0.1*tf.reduce_sum(l1_loss) / tf.maximum(1.0, assigned_boxes)

	return class_loss
