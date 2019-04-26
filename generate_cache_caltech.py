import os
import cPickle
import numpy as np
import matplotlib.pyplot as plt

root_dir = 'data/caltech/train_3'
# root_dir = 'data/caltech/test'
all_img_path = os.path.join(root_dir, 'images')
all_anno_path = os.path.join(root_dir, 'annotations_new/')
res_path_gt = 'data/cache/caltech/train_gt'
res_path_nogt = 'data/cache/caltech/train_nogt'

rows, cols = 480, 640
image_data_gt, image_data_nogt = [], []

valid_count = 0
iggt_count = 0
box_count = 0
files = sorted(os.listdir(all_anno_path))
for l in range(len(files)):
	gtname = files[l]
	imgname = files[l].split('.')[0]+'.jpg'
	img_path = os.path.join(all_img_path, imgname)
	gt_path = os.path.join(all_anno_path, gtname)

	boxes = []
	ig_boxes = []
	with open(gt_path, 'rb') as fid:
		lines = fid.readlines()
	if len(lines)>1:
		for i in range(1, len(lines)):
			info = lines[i].strip().split(' ')
			label = info[0]
			occ, ignore = info[5], info[10]
			x1, y1 = max(int(float(info[1])), 0), max(int(float(info[2])), 0)
			w, h = min(int(float(info[3])), cols - x1 - 1), min(int(float(info[4])), rows - y1 - 1)
			box = np.array([int(x1), int(y1), int(x1) + int(w), int(y1) + int(h)])
			if int(ignore) == 0:
				boxes.append(box)
			else:
				ig_boxes.append(box)
	boxes = np.array(boxes)
	ig_boxes = np.array(ig_boxes)

	annotation = {}
	annotation['filepath'] = img_path
	box_count += len(boxes)
	iggt_count += len(ig_boxes)
	annotation['bboxes'] = boxes
	annotation['ignoreareas'] = ig_boxes
	if len(boxes) == 0:
		image_data_nogt.append(annotation)
	else:
		image_data_gt.append(annotation)
		valid_count += 1
print '{} images and {} valid images, {} valid gt and {} ignored gt'.format(len(files), valid_count, box_count, iggt_count)

if not os.path.exists(res_path_gt):
	with open(res_path_gt, 'wb') as fid:
		cPickle.dump(image_data_gt, fid, cPickle.HIGHEST_PROTOCOL)
if not os.path.exists(res_path_nogt):
	with open(res_path_nogt, 'wb') as fid:
		cPickle.dump(image_data_nogt, fid, cPickle.HIGHEST_PROTOCOL)