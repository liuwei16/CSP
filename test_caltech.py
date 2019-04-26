from __future__ import division
import os
import time
import cPickle
from keras.layers import Input
from keras.models import Model
from keras_csp import config, bbox_process
from keras_csp.utilsfunc import *
from keras_csp import resnet50 as nn

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
C = config.Config()
C.offset = True
cache_path = 'data/cache/caltech/test'
with open(cache_path, 'rb') as fid:
	val_data = cPickle.load(fid)
num_imgs = len(val_data)
print 'num of val samples: {}'.format(num_imgs)

C.size_test = (480, 640)
input_shape_img = (C.size_test[0], C.size_test[1], 3)

img_input = Input(shape=input_shape_img)

# define the network prediction
preds = nn.nn_p3p4p5(img_input, offset=C.offset, num_scale=C.num_scale, trainable=True)
model = Model(img_input, preds)

if C.offset:
    w_path = 'output/valmodels/caltech/%s/off' % (C.scale)
    out_path = 'output/valresults/caltech/%s/off' % (C.scale)
else:
    w_path = 'output/valmodels/caltech/%s/nooff' % (C.scale)
    out_path = 'output/valresults/caltech/%s/nooff' % (C.scale)

if not os.path.exists(out_path):
	os.makedirs(out_path)
files = sorted(os.listdir(w_path))
for w_ind in range(51, 121):
	for f in files:
		if f.split('_')[0] == 'net' and int(f.split('_')[1][1:]) == w_ind:
			cur_file = f
			break
	weight1 = os.path.join(w_path, cur_file)
	print 'load weights from {}'.format(weight1)
	model.load_weights(weight1, by_name=True)
	res_path = os.path.join(out_path, '%03d'%int(str(w_ind)))

	print res_path
	if not os.path.exists(res_path):
		os.mkdir(res_path)
	for st in range(6, 11):
		set_path = os.path.join(res_path, 'set' + '%02d' % st)
		if not os.path.exists(set_path):
			os.mkdir(set_path)

	start_time = time.time()
	for f in range(num_imgs):
		filepath = val_data[f]['filepath']
		filepath_next = val_data[f + 1]['filepath'] if f < num_imgs - 1 else val_data[f]['filepath']
		set = filepath.split('/')[-1].split('_')[0]
		video = filepath.split('/')[-1].split('_')[1]
		frame_number = int(filepath.split('/')[-1].split('_')[2][1:6]) + 1
		frame_number_next = int(filepath_next.split('/')[-1].split('_')[2][1:6]) + 1
		set_path = os.path.join(res_path, set)
		video_path = os.path.join(set_path, video + '.txt')
		if os.path.exists(video_path):
			continue
		if frame_number == 30:
			res_all = []
		img = cv2.imread(filepath)
		x_rcnn = format_img(img, C)
		Y = model.predict(x_rcnn)

		if C.offset:
			boxes = bbox_process.parse_det_offset(Y, C, score=0.01,down=4)
		else:
			boxes = bbox_process.parse_det(Y, C, score=0.01, down=4, scale=C.scale)

		if len(boxes)>0:
			f_res = np.repeat(frame_number, len(boxes), axis=0).reshape((-1, 1))
			boxes[:, [2, 3]] -= boxes[:, [0, 1]]
			res_all += np.concatenate((f_res, boxes), axis=-1).tolist()
		if frame_number_next == 30 or f == num_imgs - 1:
			np.savetxt(video_path, np.array(res_all), fmt='%6f')
	print time.time() - start_time
