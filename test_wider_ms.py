from __future__ import division
import os
import time
import cPickle
from keras.layers import Input
from keras.models import Model
from keras_csp import config, bbox_process
from keras_csp.utilsfunc import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
C = config.Config()
C.offset = True
C.scale = 'hw'
C.num_scale = 2
cache_path = 'data/cache/widerface/val'
with open(cache_path, 'rb') as fid:
	val_data = cPickle.load(fid)
num_imgs = len(val_data)
print 'num of val samples: {}'.format(num_imgs)

C.size_test = [0, 0]
input_shape_img = (None, None, 3)
img_input = Input(shape=input_shape_img)

# define the base network (resnet here, can be MobileNet, etc)
from keras_csp import resnet50 as nn
# define the network prediction
preds = nn.nn_p3p4p5(img_input, offset=C.offset, num_scale=C.num_scale, trainable=True)
model = Model(img_input, preds)

if C.offset:
    w_path = 'output/valmodels/wider/%s/off' % (C.scale)
    out_path = 'output/valresults/wider/%s/off' % (C.scale)
else:
    w_path = 'output/valmodels/wider/%s/nooff' % (C.scale)
    out_path = 'output/valresults/wider/%s/nooff' % (C.scale)
if not os.path.exists(out_path):
	os.makedirs(out_path)
files = sorted(os.listdir(w_path))
# get the results from epoch 51 to epoch 150
for w_ind in range(382,383):
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

	start_time = time.time()
	for f in range(num_imgs):
		filepath = val_data[f]['filepath']
		event = filepath.split('/')[-2]
		event_path = os.path.join(res_path, event)
		if not os.path.exists(event_path):
			os.mkdir(event_path)
		filename = filepath.split('/')[-1].split('.')[0]
		txtpath = os.path.join(event_path, filename + '.txt')
		if os.path.exists(txtpath):
			continue

		img = cv2.imread(filepath)

		def detect_face(img, scale=1, flip=False):
			img_h, img_w = img.shape[:2]
			img_h_new, img_w_new = int(np.ceil(scale * img_h / 16) * 16), int(np.ceil(scale * img_w / 16) * 16)
			scale_h, scale_w = img_h_new / img_h, img_w_new / img_w

			img_s = cv2.resize(img, None, None, fx=scale_w, fy=scale_h, interpolation=cv2.INTER_LINEAR)
			# img_h, img_w = img_s.shape[:2]
			# print frame_number
			C.size_test[0] = img_h_new
			C.size_test[1] = img_w_new

			if flip:
				img_sf = cv2.flip(img_s, 1)
				# x_rcnn = format_img_pad(img_sf, C)
				x_rcnn = format_img(img_sf, C)
			else:
				# x_rcnn = format_img_pad(img_s, C)
				x_rcnn = format_img(img_s, C)
			Y = model.predict(x_rcnn)
			boxes = bbox_process.parse_wider_offset(Y, C, score=0.05, nmsthre=0.6)
			if len(boxes) > 0:
				keep_index = np.where(np.minimum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]) >= 12)[0]
				boxes = boxes[keep_index, :]
			if len(boxes) > 0:
				if flip:
					boxes[:, [0, 2]] = img_s.shape[1] - boxes[:, [2, 0]]
				boxes[:, 0:4:2] = boxes[:, 0:4:2] / scale_w
				boxes[:, 1:4:2] = boxes[:, 1:4:2] / scale_h
			else:
				boxes = np.empty(shape=[0, 5], dtype=np.float32)
			return boxes

		def im_det_ms_pyramid(image, max_im_shrink):
			# shrink detecting and shrink only detect big face
			det_s = np.row_stack((detect_face(image, 0.5), detect_face(image, 0.5, flip=True)))
			index = np.where(np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 64)[0]
			det_s = det_s[index, :]

			det_temp = np.row_stack((detect_face(image, 0.75), detect_face(image, 0.75, flip=True)))
			index = np.where(np.maximum(det_temp[:, 2] - det_temp[:, 0] + 1, det_temp[:, 3] - det_temp[:, 1] + 1) > 32)[0]
			det_temp = det_temp[index, :]
			det_s = np.row_stack((det_s, det_temp))

			det_temp = np.row_stack((detect_face(image, 0.25), detect_face(image, 0.25, flip=True)))
			index = np.where(np.maximum(det_temp[:, 2] - det_temp[:, 0] + 1, det_temp[:, 3] - det_temp[:, 1] + 1) > 96)[0]
			det_temp = det_temp[index, :]
			det_s = np.row_stack((det_s, det_temp))

			st = [1.25, 1.5, 1.75, 2.0, 2.25]
			for i in range(len(st)):
				if (st[i] <= max_im_shrink):
					det_temp = np.row_stack((detect_face(image, st[i]), detect_face(image, st[i], flip=True)))
					# Enlarged images are only used to detect small faces.
					if st[i] == 1.25:
						index = np.where(
							np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1, det_temp[:, 3] - det_temp[:, 1] + 1) < 128)[0]
						det_temp = det_temp[index, :]
					elif st[i] == 1.5:
						index = np.where(
							np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1, det_temp[:, 3] - det_temp[:, 1] + 1) < 96)[0]
						det_temp = det_temp[index, :]
					elif st[i] == 1.75:
						index = np.where(
							np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1, det_temp[:, 3] - det_temp[:, 1] + 1) < 64)[0]
						det_temp = det_temp[index, :]
					elif st[i] == 2.0:
						index = np.where(
							np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1, det_temp[:, 3] - det_temp[:, 1] + 1) < 48)[0]
						det_temp = det_temp[index, :]
					elif st[i] == 2.25:
						index = np.where(
							np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1, det_temp[:, 3] - det_temp[:, 1] + 1) < 32)[0]
						det_temp = det_temp[index, :]
					det_s = np.row_stack((det_s, det_temp))
			return det_s

		max_im_shrink = (0x7fffffff / 577.0 / (img.shape[0] * img.shape[1])) ** 0.5  # the max size of input image
		shrink = max_im_shrink if max_im_shrink < 1 else 1
		det0 = detect_face(img)
		det1 = detect_face(img, flip=True)
		det2 = im_det_ms_pyramid(img, max_im_shrink)
		# merge all test results via bounding box voting
		det = np.row_stack((det0, det1, det2))
		keep_index = np.where(np.minimum(det[:, 2] - det[:, 0], det[:, 3] - det[:, 1]) >= 3)[0]
		det = det[keep_index, :]
		dets = bbox_process.soft_bbox_vote(det, thre=0.4)
		keep_index = np.where((dets[:, 2] - dets[:, 0] + 1) * (dets[:, 3] - dets[:, 1] + 1) >= 6 ** 2)[0]
		dets = dets[keep_index, :]

		with open(txtpath, 'w') as f:
			f.write('{:s}\n'.format(filename))
			f.write('{:d}\n'.format(len(dets)))
			for line in dets:
				f.write('{:.0f} {:.0f} {:.0f} {:.0f} {:.3f}\n'.
						format(line[0], line[1], line[2] - line[0] + 1, line[3] - line[1] + 1, line[4]))
	print time.time() - start_time