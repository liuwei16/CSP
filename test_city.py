from __future__ import division
import os
import time
import cPickle
from keras.layers import Input
from keras.models import Model
from keras_csp import config, bbox_process
from keras_csp.utilsfunc import *

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
C = config.Config()
C.offset = True
cache_path = 'data/cache/cityperson/val_500'
with open(cache_path, 'rb') as fid:
	val_data = cPickle.load(fid)
num_imgs = len(val_data)
print 'num of val samples: {}'.format(num_imgs)

C.size_test = (1024, 2048)
input_shape_img = (C.size_test[0], C.size_test[1], 3)
img_input = Input(shape=input_shape_img)

# define the base network (resnet here, can be MobileNet, etc)
if C.network=='resnet50':
    from keras_csp import resnet50 as nn
elif C.network=='mobilenet':
    from keras_csp import mobilenet as nn
else:
    raise NotImplementedError('Not support network: {}'.format(C.network))

# define the network prediction
preds = nn.nn_p3p4p5(img_input, offset=C.offset, num_scale=C.num_scale, trainable=True)
model = Model(img_input, preds)

if C.offset:
    w_path = 'output/valmodels/city/%s/off' % (C.scale)
    out_path = 'output/valresults/city/%s/off' % (C.scale)
else:
    w_path = 'output/valmodels/city/%s/nooff' % (C.scale)
    out_path = 'output/valresults/city/%s/nooff' % (C.scale)
if not os.path.exists(out_path):
	os.makedirs(out_path)
files = sorted(os.listdir(w_path))
# get the results from epoch 51 to epoch 150
for w_ind in range(51,151):
	for f in files:
		if f.split('_')[0] == 'net' and int(f.split('_')[1][1:]) == w_ind:
			cur_file = f
			break
	weight1 = os.path.join(w_path, cur_file)
	print 'load weights from {}'.format(weight1)
	model.load_weights(weight1, by_name=True)
	res_path = os.path.join(out_path, '%03d'%int(str(w_ind)))
	if not os.path.exists(res_path):
		os.makedirs(res_path)
	print res_path
	res_file = os.path.join(res_path, 'val_det.txt')
	res_all = []
	start_time = time.time()
	for f in range(num_imgs):
		filepath = val_data[f]['filepath']
		img = cv2.imread(filepath)
		x_rcnn = format_img(img, C)
		Y = model.predict(x_rcnn)

		if C.offset:
			boxes = bbox_process.parse_det_offset(Y, C, score=0.1,down=4)
		else:
			boxes = bbox_process.parse_det(Y, C, score=0.1, down=4, scale=C.scale)
		if len(boxes)>0:
			f_res = np.repeat(f+1, len(boxes), axis=0).reshape((-1, 1))
			boxes[:, [2, 3]] -= boxes[:, [0, 1]]
			res_all += np.concatenate((f_res, boxes), axis=-1).tolist()
	np.savetxt(res_file, np.array(res_all), fmt='%6f')
	print time.time() - start_time
