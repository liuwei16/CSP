from __future__ import absolute_import
from __future__ import division
# import numpy as np
# import cv2
import random
from . import data_augment
from .bbox_transform import *

def calc_gt_center(C, img_data,r=2, down=4,scale='h',offset=True):
	def gaussian(kernel):
		sigma = ((kernel-1) * 0.5 - 1) * 0.3 + 0.8
		s = 2*(sigma**2)
		dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
		return np.reshape(dx,(-1,1))
	gts = np.copy(img_data['bboxes'])
	igs = np.copy(img_data['ignoreareas'])
	scale_map = np.zeros((int(C.size_train[0]/down), int(C.size_train[1]/down), 2))
	if scale=='hw':
		scale_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
	if offset:
		offset_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
	seman_map = np.zeros((int(C.size_train[0]/down), int(C.size_train[1]/down), 3))
	seman_map[:,:,1] = 1
	if len(igs) > 0:
		igs = igs/down
		for ind in range(len(igs)):
			x1,y1,x2,y2 = int(igs[ind,0]), int(igs[ind,1]), int(np.ceil(igs[ind,2])), int(np.ceil(igs[ind,3]))
			seman_map[y1:y2, x1:x2,1] = 0
	if len(gts)>0:
		gts = gts/down
		for ind in range(len(gts)):
			# x1, y1, x2, y2 = int(round(gts[ind, 0])), int(round(gts[ind, 1])), int(round(gts[ind, 2])), int(round(gts[ind, 3]))
			x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
			c_x, c_y = int((gts[ind, 0] + gts[ind, 2]) / 2), int((gts[ind, 1] + gts[ind, 3]) / 2)
			dx = gaussian(x2-x1)
			dy = gaussian(y2-y1)
			gau_map = np.multiply(dy, np.transpose(dx))
			seman_map[y1:y2, x1:x2,0] = np.maximum(seman_map[y1:y2, x1:x2,0], gau_map)
			seman_map[y1:y2, x1:x2,1] = 1
			seman_map[c_y, c_x, 2] = 1

			if scale == 'h':
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 1] = 1
			elif scale=='w':
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 0] = np.log(gts[ind, 2] - gts[ind, 0])
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 1] = 1
			elif scale=='hw':
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 1] = np.log(gts[ind, 2] - gts[ind, 0])
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 2] = 1
			if offset:
				offset_map[c_y, c_x, 0] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - 0.5
				offset_map[c_y, c_x, 1] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - 0.5
				offset_map[c_y, c_x, 2] = 1

	if offset:
		return seman_map,scale_map,offset_map
	else:
		return seman_map, scale_map

def calc_gt_top(C, img_data,r=2):
	def gaussian(kernel):
		sigma = ((kernel-1) * 0.5 - 1) * 0.3 + 0.8
		s = 2*(sigma**2)
		dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
		return np.reshape(dx,(-1,1))
	gts = np.copy(img_data['bboxes'])
	igs = np.copy(img_data['ignoreareas'])
	scale_map = np.zeros((int(C.size_train[0]/4), int(C.size_train[1]/4), 2))
	seman_map = np.zeros((int(C.size_train[0]/4), int(C.size_train[1]/4), 3))
	seman_map[:,:,1] = 1
	if len(igs) > 0:
		igs = igs/4
		for ind in range(len(igs)):
			x1,y1,x2,y2 = int(igs[ind,0]), int(igs[ind,1]), int(np.ceil(igs[ind,2])), int(np.ceil(igs[ind,3]))
			seman_map[y1:y2, x1:x2,1] = 0
	if len(gts)>0:
		gts = gts/4
		for ind in range(len(gts)):
			x1, y1, x2, y2 = int(round(gts[ind, 0])), int(round(gts[ind, 1])), int(round(gts[ind, 2])), int(round(gts[ind, 3]))
			w = x2-x1
			c_x = int((gts[ind, 0] + gts[ind, 2]) / 2)

			dx = gaussian(w)
			dy = gaussian(w)
			gau_map = np.multiply(dy, np.transpose(dx))

			ty = np.maximum(0,int(round(y1-w/2)))
			ot = ty-int(round(y1-w/2))
			seman_map[ty:ty+w-ot, x1:x2,0] = np.maximum(seman_map[ty:ty+w-ot, x1:x2,0], gau_map[ot:,:])
			seman_map[ty:ty+w-ot, x1:x2,1] = 1
			seman_map[y1, c_x, 2] = 1

			scale_map[y1-r:y1+r+1, c_x-r:c_x+r+1, 0] = np.log(gts[ind,3]-gts[ind,1])
			scale_map[y1-r:y1+r+1, c_x-r:c_x+r+1, 1] = 1
	return seman_map,scale_map

def calc_gt_bottom(C, img_data, r=2):
	def gaussian(kernel):
		sigma = ((kernel-1) * 0.5 - 1) * 0.3 + 0.8
		s = 2*(sigma**2)
		dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
		return np.reshape(dx,(-1,1))
	gts = np.copy(img_data['bboxes'])
	igs = np.copy(img_data['ignoreareas'])
	scale_map = np.zeros((int(C.size_train[0]/4), int(C.size_train[1]/4), 2))
	seman_map = np.zeros((int(C.size_train[0]/4), int(C.size_train[1]/4), 3))
	seman_map[:,:,1] = 1
	if len(igs) > 0:
		igs = igs/4
		for ind in range(len(igs)):
			x1,y1,x2,y2 = int(igs[ind,0]), int(igs[ind,1]), int(np.ceil(igs[ind,2])), int(np.ceil(igs[ind,3]))
			seman_map[y1:y2, x1:x2,1] = 0
	if len(gts)>0:
		gts = gts/4
		for ind in range(len(gts)):
			x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
			y2 = np.minimum(int(C.random_crop[0] / 4) - 1, y2)
			w = x2 - x1
			c_x = int((gts[ind, 0] + gts[ind, 2]) / 2)
			dx = gaussian(w)
			dy = gaussian(w)
			gau_map = np.multiply(dy, np.transpose(dx))

			by = np.minimum(int(C.random_crop[0]/4)-1, int(round(y2+w/2)))
			ob = int(round(y2+w/2))-by
			seman_map[by-w+ob:by, x1:x2, 0] = np.maximum(seman_map[by-w+ob:by, x1:x2, 0], gau_map[:w-ob, :])
			seman_map[by-w+ob:by, x1:x2, 1] = 1
			seman_map[y2, c_x, 2] = 1

			scale_map[y2-r:y2+r+1, c_x-r:c_x+r+1, 0] = np.log(gts[ind,3]-gts[ind,1])
			scale_map[y2-r:y2+r+1, c_x-r:c_x+r+1, 1] = 1

	return seman_map,scale_map

def get_data(ped_data, C, batchsize = 8):
	current_ped = 0
	while True:
		x_img_batch, y_seman_batch, y_height_batch, y_offset_batch = [], [], [], []
		if current_ped>len(ped_data)-batchsize:
			random.shuffle(ped_data)
			current_ped = 0
		for img_data in ped_data[current_ped:current_ped + batchsize]:
			try:
				img_data, x_img = data_augment.augment(img_data, C)
				if C.offset:
					y_seman, y_height, y_offset = calc_gt_center(C, img_data, down=C.down, scale=C.scale, offset=True)
				else:
					if C.point == 'top':
						y_seman, y_height = calc_gt_top(C, img_data)
					elif C.point == 'bottom':
						y_seman, y_height = calc_gt_bottom(C, img_data)
					else:
						y_seman, y_height = calc_gt_center(C, img_data,down=C.down, scale=C.scale, offset=False)

				x_img = x_img.astype(np.float32)
				x_img[:, :, 0] -= C.img_channel_mean[0]
				x_img[:, :, 1] -= C.img_channel_mean[1]
				x_img[:, :, 2] -= C.img_channel_mean[2]

				x_img_batch.append(np.expand_dims(x_img, axis=0))
				y_seman_batch.append(np.expand_dims(y_seman, axis=0))
				y_height_batch.append(np.expand_dims(y_height, axis=0))
				if C.offset:
					y_offset_batch.append(np.expand_dims(y_offset, axis=0))
			except Exception as e:
				print ('get_batch_gt:',e)
		x_img_batch = np.concatenate(x_img_batch,axis=0)
		y_seman_batch = np.concatenate(y_seman_batch, axis=0)
		y_height_batch = np.concatenate(y_height_batch, axis=0)
		if C.offset:
			y_offset_batch = np.concatenate(y_offset_batch, axis=0)
		current_ped += batchsize
		if C.offset:
			yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch), np.copy(y_offset_batch)]
		else:
			yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch)]

def get_data_hybrid(ped_data, emp_data, C, batchsize = 8,hyratio=0.5):
	current_ped = 0
	current_emp = 0
	batchsize_ped = int(batchsize * hyratio)
	batchsize_emp = batchsize - batchsize_ped
	while True:
		x_img_batch, y_seman_batch, y_height_batch, y_offset_batch = [], [], [], []
		if current_ped>len(ped_data)-batchsize_ped:
			random.shuffle(ped_data)
			current_ped = 0
		if current_emp>len(emp_data)-batchsize_emp:
			random.shuffle(emp_data)
			current_emp = 0
		for img_data in ped_data[current_ped:current_ped + batchsize_ped]:
			try:
				img_data, x_img = data_augment.augment(img_data, C)
				if C.offset:
					y_seman, y_height, y_offset = calc_gt_center(C, img_data, down=C.down, scale=C.scale, offset=C.offset)
				else:
					if C.point == 'top':
						y_seman, y_height = calc_gt_top(C, img_data)
					elif C.point == 'bottom':
						y_seman, y_height = calc_gt_bottom(C, img_data)
					else:
						y_seman, y_height = calc_gt_center(C, img_data,down=C.down, scale=C.scale, offset=False)

				x_img = x_img.astype(np.float32)
				x_img[:, :, 0] -= C.img_channel_mean[0]
				x_img[:, :, 1] -= C.img_channel_mean[1]
				x_img[:, :, 2] -= C.img_channel_mean[2]

				x_img_batch.append(np.expand_dims(x_img, axis=0))
				y_seman_batch.append(np.expand_dims(y_seman, axis=0))
				y_height_batch.append(np.expand_dims(y_height, axis=0))
				if C.offset:
					y_offset_batch.append(np.expand_dims(y_offset, axis=0))

			except Exception as e:
				print ('get_batch_gt:',e)
		for img_data in emp_data[current_emp:current_emp + batchsize_emp]:
			try:
				img_data, x_img = data_augment.augment(img_data, C)
				if C.offset:
					y_seman, y_height, y_offset = calc_gt_center(C, img_data, down=C.down, scale=C.scale, offset=C.offset)
				else:
					if C.point == 'top':
						y_seman, y_height = calc_gt_top(C, img_data)
					elif C.point == 'bottom':
						y_seman, y_height = calc_gt_bottom(C, img_data)
					else:
						y_seman, y_height = calc_gt_center(C, img_data,down=C.down, scale=C.scale, offset=False)

				x_img = x_img.astype(np.float32)
				x_img[:, :, 0] -= C.img_channel_mean[0]
				x_img[:, :, 1] -= C.img_channel_mean[1]
				x_img[:, :, 2] -= C.img_channel_mean[2]

				x_img_batch.append(np.expand_dims(x_img, axis=0))
				y_seman_batch.append(np.expand_dims(y_seman, axis=0))
				y_height_batch.append(np.expand_dims(y_height, axis=0))
				if C.offset:
					y_offset_batch.append(np.expand_dims(y_offset, axis=0))
			except Exception as e:
				print ('get_batch_gt_emp:',e)
		x_img_batch = np.concatenate(x_img_batch,axis=0)
		y_seman_batch = np.concatenate(y_seman_batch, axis=0)
		y_height_batch = np.concatenate(y_height_batch, axis=0)
		if C.offset:
			y_offset_batch = np.concatenate(y_offset_batch, axis=0)
		current_ped += batchsize_ped
		current_emp += batchsize_emp
		if C.offset:
			yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch), np.copy(y_offset_batch)]
		else:
			yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch)]

def get_data_wider(ped_data, C, batchsize = 8):
	current_ped = 0
	while True:
		x_img_batch, y_seman_batch, y_height_batch, y_offset_batch = [], [], [], []
		if current_ped>len(ped_data)-batchsize:
			random.shuffle(ped_data)
			current_ped = 0
		for img_data in ped_data[current_ped:current_ped + batchsize]:
			try:
				img_data, x_img = data_augment.augment_wider(img_data, C)
				if C.offset:
					y_seman, y_height, y_offset = calc_gt_center(C, img_data, down=C.down, scale=C.scale, offset=True)
				else:
					y_seman, y_height = calc_gt_center(C, img_data,down=C.down, scale=C.scale, offset=False)

				x_img = x_img.astype(np.float32)
				x_img[:, :, 0] -= C.img_channel_mean[0]
				x_img[:, :, 1] -= C.img_channel_mean[1]
				x_img[:, :, 2] -= C.img_channel_mean[2]

				x_img_batch.append(np.expand_dims(x_img, axis=0))
				y_seman_batch.append(np.expand_dims(y_seman, axis=0))
				y_height_batch.append(np.expand_dims(y_height, axis=0))
				if C.offset:
					y_offset_batch.append(np.expand_dims(y_offset, axis=0))
			except Exception as e:
				print ('get_batch_gt:',e)
		x_img_batch = np.concatenate(x_img_batch,axis=0)
		y_seman_batch = np.concatenate(y_seman_batch, axis=0)
		y_height_batch = np.concatenate(y_height_batch, axis=0)
		if C.offset:
			y_offset_batch = np.concatenate(y_offset_batch, axis=0)
		current_ped += batchsize
		if C.offset:
			yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch), np.copy(y_offset_batch)]
		else:
			yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch)]
