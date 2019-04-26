from __future__ import division
import random
import sys, os
import time
import numpy as np
import cPickle
from keras.utils import generic_utils
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from keras_csp import config, data_generators
from keras_csp import losses as losses

# get the config parameters
C = config.Config()
C.gpu_ids = '0,1,2,3,4,5,6,7'
C.onegpu = 4
C.size_train = (704,704)
C.init_lr = 2e-4
C.offset = True
C.scale = 'hw'
C.num_scale = 2
C.num_epochs = 400
C.iter_per_epoch = 4000
num_gpu = len(C.gpu_ids.split(','))
batchsize = C.onegpu * num_gpu
os.environ["CUDA_VISIBLE_DEVICES"] = C.gpu_ids

# get the training data
cache_path = 'data/cache/widerface/train'
with open(cache_path, 'rb') as fid:
    train_data = cPickle.load(fid)
num_imgs_train = len(train_data)
print 'num of training samples: {}'.format(num_imgs_train)
data_gen_train = data_generators.get_data_wider(train_data, C, batchsize=batchsize)

# define the base network (resnet here, can be MobileNet, etc)
if C.network=='resnet50':
    from keras_csp import resnet50 as nn
    weight_path = 'data/models/resnet50_weights_tf_dim_ordering_tf_kernels.h5'

input_shape_img = (C.size_train[0], C.size_train[1], 3)
img_input = Input(shape=input_shape_img)
# define the network prediction
preds = nn.nn_p3p4p5(img_input, offset=C.offset, num_scale=C.num_scale, trainable=True)
preds_tea = nn.nn_p3p4p5(img_input, offset=C.offset, num_scale=C.num_scale, trainable=True)

model = Model(img_input, preds)
if num_gpu>1:
    from keras_csp.parallel_model import ParallelModel
    model = ParallelModel(model, int(num_gpu))
    model_stu = Model(img_input, preds)
model_tea = Model(img_input, preds_tea)

model.load_weights(weight_path, by_name=True)
model_tea.load_weights(weight_path, by_name=True)
print 'load weights from {}'.format(weight_path)

if C.offset:
    out_path = 'output/valmodels/wider/%s/off' % (C.scale)
else:
    out_path = 'output/valmodels/wider/%s/nooff' % (C.scale)
if not os.path.exists(out_path):
    os.makedirs(out_path)
res_file = os.path.join(out_path,'records.txt')

optimizer = Adam(lr=C.init_lr)
if C.offset:
    model.compile(optimizer=optimizer, loss=[losses.cls_center, losses.regr_hw, losses.regr_offset])
else:
    model.compile(optimizer=optimizer, loss=[losses.cls_center, losses.regr_hw])

epoch_length = int(C.iter_per_epoch/batchsize)
iter_num = 0
add_epoch = 0
losses = np.zeros((epoch_length, 3))

best_loss = np.Inf
print('Starting training with lr {} and alpha {}'.format(C.init_lr, C.alpha))
start_time = time.time()
total_loss_r, cls_loss_r1, regr_loss_r1, offset_loss_r1 = [], [], [], []
for epoch_num in range(C.num_epochs):
    progbar = generic_utils.Progbar(epoch_length)
    print('Epoch {}/{}'.format(epoch_num + 1 + add_epoch, C.num_epochs + C.add_epoch))
    while True:
        try:
            X, Y = next(data_gen_train)
            loss_s1 = model.train_on_batch(X, Y)

            for l in model_tea.layers:
                weights_tea = l.get_weights()
                if len(weights_tea)>0:
                    if num_gpu > 1:
                        weights_stu = model_stu.get_layer(name=l.name).get_weights()
                    else:
                        weights_stu = model.get_layer(name=l.name).get_weights()
                    weights_tea = [C.alpha*w_tea + (1-C.alpha)*w_stu for (w_tea, w_stu) in zip(weights_tea, weights_stu)]
                    l.set_weights(weights_tea)
            # print loss_s1
            losses[iter_num, 0] = loss_s1[1]
            losses[iter_num, 1] = loss_s1[2]
            if C.offset:
                losses[iter_num, 2] = loss_s1[3]
            else:
                losses[iter_num, 2] = 0

            iter_num += 1
            if iter_num % 20 == 0:
                progbar.update(iter_num,
                               [('cls', np.mean(losses[:iter_num, 0])), ('regr_h', np.mean(losses[:iter_num, 1])), ('offset', np.mean(losses[:iter_num, 2]))])
            if iter_num == epoch_length:
                cls_loss1 = np.mean(losses[:, 0])
                regr_loss1 = np.mean(losses[:, 1])
                offset_loss1 = np.mean(losses[:, 2])
                total_loss = cls_loss1+regr_loss1+offset_loss1

                total_loss_r.append(total_loss)
                cls_loss_r1.append(cls_loss1)
                regr_loss_r1.append(regr_loss1)
                offset_loss_r1.append(offset_loss1)
                print('Total loss: {}'.format(total_loss))
                print('Elapsed time: {}'.format(time.time() - start_time))

                iter_num = 0
                start_time = time.time()

                if total_loss < best_loss:
                    print('Total loss decreased from {} to {}, saving weights'.format(best_loss, total_loss))
                    best_loss = total_loss
                model_tea.save_weights(os.path.join(out_path, 'net_e{}_l{}.hdf5'.format(epoch_num + 1 + add_epoch, total_loss)))
                break
        except Exception as e:
            print ('Exception: {}'.format(e))
            continue
    records = np.concatenate((np.asarray(total_loss_r).reshape((-1, 1)),
                              np.asarray(cls_loss_r1).reshape((-1, 1)),
                              np.asarray(regr_loss_r1).reshape((-1, 1)),
                              np.asarray(offset_loss_r1).reshape((-1, 1)),),
                             axis=-1)
    np.savetxt(res_file, np.array(records), fmt='%.6f')
print('Training complete, exiting.')