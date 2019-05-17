from __future__ import print_function

caffe_root = '/home/terry/software/caffe-ssd-gpu'
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')
import caffe
from caffe import layers as L
from caffe import params as P


def conv_layer(net, from_layer, out_layer, use_relu, num_output,
               kernel_size, pad, stride, lr_mult=1, use_group=False,
               conv_prefix='', conv_postfix=''):
    kwargs = {
        'param': [
            dict(lr_mult=lr_mult, decay_mult=lr_mult),
            dict(lr_mult=2 * lr_mult, decay_mult=0)],
        'weight_filler': dict(type='gaussian', std=0.01),
        'bias_filler': dict(type='constant', value=0)
    }

    conv_name = '{}{}{}'.format(conv_prefix, out_layer, conv_postfix)
    if use_group:
        net[conv_name] = L.Convolution(net[from_layer], num_output=num_output, group=2,
                                       kernel_size=kernel_size, pad=pad, stride=stride, **kwargs)
    else:
        net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
                                       kernel_size=kernel_size, pad=pad, stride=stride, **kwargs)

    if use_relu:
        relu_name = '{}/relu'.format(conv_name)
        net[relu_name] = L.ReLU(net[conv_name], in_place=True)


def inner_product_layer(net, from_layer, out_layer, use_relu, num_output, lr_mult=1,
                        product_prefix='', product_postfix=''):
    kwargs = {
        'param': [
            dict(lr_mult=lr_mult, decay_mult=lr_mult),
            dict(lr_mult=2 * lr_mult, decay_mult=0)],
        'weight_filler': dict(type='gaussian'),
        'bias_filler': dict(type='constant')
    }

    product_name = '{}{}{}'.format(product_prefix, out_layer, product_postfix)
    net[product_name] = L.InnerProduct(net[from_layer], num_output=num_output, **kwargs)
    if use_relu:
        relu_name = '{}/relu'.format(product_name)
        net[relu_name] = L.ReLU(net[product_name], in_place=True)


def vgg_body(net, from_layer, vgg_version='16'):
    conv_layer(net, from_layer, 'conv1_1', use_relu=True,
               num_output=64, kernel_size=3, pad=1, stride=1)
    conv_layer(net, net.keys()[-1], 'conv1_2', use_relu=True,
               num_output=64, kernel_size=3, pad=1, stride=1)

    net.pool1 = L.Pooling(net[net.keys()[-1]], kernel_size=2, stride=2, pool=P.Pooling.MAX)

    conv_layer(net, net.keys()[-1], 'conv2_1', use_relu=True,
               num_output=128, kernel_size=3, pad=1, stride=1)
    conv_layer(net, net.keys()[-1], 'conv2_2', use_relu=True,
               num_output=128, kernel_size=3, pad=1, stride=1)

    net.pool2 = L.Pooling(net[net.keys()[-1]], kernel_size=2, stride=2, pool=P.Pooling.MAX)

    if vgg_version == '16':
        for i in range(1, 3):
            conv_layer(net, net.keys()[-1], 'conv3_{}'.format(i), use_relu=True,
                       num_output=256, kernel_size=3, pad=1, stride=1)
    else:   # vgg19
        for i in range(1, 4):
            conv_layer(net, net.keys()[-1], 'conv3_{}'.format(i), use_relu=True,
                       num_output=256, kernel_size=3, pad=1, stride=1)

    net.pool3 = L.Pooling(net[net.keys()[-1]], kernel_size=2, stride=2, pool=P.Pooling.MAX)

    if vgg_version == '16':
        for i in range(1, 3):
            conv_layer(net, net.keys()[-1], 'conv4_{}'.format(i), use_relu=True,
                       num_output=512, kernel_size=3, pad=1, stride=1)
    else:   # vgg19
        for i in range(1, 4):
            conv_layer(net, net.keys()[-1], 'conv4_{}'.format(i), use_relu=True,
                       num_output=512, kernel_size=3, pad=1, stride=1)

    net.pool4 = L.Pooling(net[net.keys()[-1]], kernel_size=2, stride=2, pool=P.Pooling.MAX)

    if vgg_version == '16':
        for i in range(1, 3):
            conv_layer(net, net.keys()[-1], 'conv5_{}'.format(i), use_relu=True,
                       num_output=512, kernel_size=3, pad=1, stride=1)
    else:   # vgg19
        for i in range(1, 4):
            conv_layer(net, net.keys()[-1], 'conv5_{}'.format(i), use_relu=True,
                       num_output=512, kernel_size=3, pad=1, stride=1)

    net.pool5 = L.Pooling(net[net.keys()[-1]], kernel_size=2, stride=2, pool=P.Pooling.MAX)

    inner_product_layer(net, net.keys()[-1], 'fc6', use_relu=True, num_output=4096)
    net.drop6 = L.Dropout(net[net.keys()[-1]], dropout_ratio=0.5, in_place=True)

    inner_product_layer(net, net.keys()[-1], 'fc7', use_relu=True, num_output=4096)
    net.drop7 = L.Dropout(net[net.keys()[-1]], dropout_ratio=0.5, in_place=True)

    inner_product_layer(net, net.keys()[-1], 'fc8', use_relu=False, num_output=1000)


if __name__ == '__main__':
    n = caffe.NetSpec()
    n.data, n.label = L.Data(data_param=dict(batch_size=256, backend=P.Data.LMDB, source='path-to-lmdb'),
                             transform_param=dict(mirror=1, crop_size=227, mean_file='imagenet_mean.binaryproto'),
                             include=dict(phase=0),     # 0 for TRAIN and 1 for TEST
                             ntop=2)

    n.data2, n.label2 = L.Data(data_param=dict(batch_size=50, backend=P.Data.LMDB, source='path-to-lmdb'),
                               transform_param=dict(mirror=0, crop_size=227, mean_file='imagenet_mean.binaryproto'),
                               include=dict(phase=1),
                               ntop=2)

    vgg_body(n, from_layer='data', vgg_version='16')
    n.accuracy = L.Accuracy(n.fc8, n.label, include=dict(phase=1))
    n.loss = L.SoftmaxWithLoss(n.fc8, n.label)

    with open('/home/terry/github/my-caffemodel-scripts/example/vgg_train.prototxt', 'w') as f:
        print('name: "{}_train_test"'.format('vgg16'), file=f)
        print(n.to_proto(), file=f)
