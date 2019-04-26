
class Config(object):
	def __init__(self):
		self.gpu_ids = '0'
		self.onegpu = 2
		self.num_epochs = 150
		self.add_epoch = 0
		self.iter_per_epoch = 2000
		self.init_lr = 1e-4
		self.alpha = 0.999

		# setting for network architechture
		self.network = 'resnet50' # or 'mobilenet'
		self.point = 'center' # or 'top', 'bottom
		self.scale = 'h' # or 'w', 'hw'
		self.num_scale = 1 # 1 for height (or width) prediction, 2 for height+width prediction
		self.offset = False # append offset prediction or not
		self.down = 4 # downsampling rate of the feature map for detection
		self.radius = 2 # surrounding areas of positives for the scale map

		# setting for data augmentation
		self.use_horizontal_flips = True
		self.brightness = (0.5, 2, 0.5)
		self.size_train = (336, 448)

		# image channel-wise mean to subtract, the order is BGR
		self.img_channel_mean = [103.939, 116.779, 123.68]
