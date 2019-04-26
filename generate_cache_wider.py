import os
import cv2
import cPickle
import numpy as np
import matplotlib.pyplot as plt

root_dir = 'data/WiderFace/'
img_path = os.path.join(root_dir, 'WIDER_train/images')
anno_path = os.path.join(root_dir, 'wider_face_split','wider_face_train_bbx_gt.txt')
# anno_path = os.path.join(root_dir, 'wider_face_split','wider_face_test_filelist.txt')

res_path = 'data/cache/train'

image_data = []
valid_count = 0
img_count = 0
box_count = 0
with open(anno_path, 'rb') as fid:
	lines = fid.readlines()
num_lines = len(lines)

index = 0
while index<num_lines:
	filename = lines[index].strip()
	img_count += 1
	if img_count%1000 == 0:
		print img_count
	num_obj = int(lines[index+1])
	filepath = os.path.join(img_path, filename)
	img = cv2.imread(filepath)
	img_height, img_width = img.shape[:2]
	boxes = []
	if num_obj>0:
		for i in range(num_obj):
			info = lines[index+2+i].strip().split(' ')
			x1, y1 = max(int(info[0]), 0), max(int(info[1]), 0)
			w, h = min(int(info[2]), img_width - x1 - 1), min(int(info[3]), img_height - y1 - 1)
			if w>=5 and h>=5:
				box = np.array([x1, y1, x1+w, y1+h])
				boxes.append(box)
	boxes = np.array(boxes)
	box_count += len(boxes)
	if len(boxes)>0:
		valid_count += 1
		annotation = {}
		annotation['filepath'] = filepath
		annotation['bboxes'] = boxes
		image_data.append(annotation)
	index += (2+num_obj)

print '{} images and {} valid images and {} boxes'.format(img_count, valid_count,box_count)
with open(res_path, 'wb') as fid:
	cPickle.dump(image_data, fid, cPickle.HIGHEST_PROTOCOL)