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
               kernel_size, pad, stride, lr_mult=1,
               conv_prefix='', conv_postfix=''):
    kwargs = {
        'param': [
            dict(lr_mult=lr_mult),
            dict(lr_mult=2 * lr_mult)],
        'weight_filler': dict(type='xavier'),
        'bias_filler': dict(type='constant', value=0)
    }

    conv_name = '{}{}{}'.format(conv_prefix, out_layer, conv_postfix)
    net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
                                   kernel_size=kernel_size, pad=pad, stride=stride, **kwargs)
    if use_relu:
        relu_name = '{}/relu'.format(conv_name)
        net[relu_name] = L.ReLU(net[conv_name], in_place=True)


def inner_product_layer(net, from_layer, out_layer, use_relu, num_output, lr_mult=1,
                        product_prefix='', product_postfix=''):
    kwargs = {
        'param': [
            dict(lr_mult=lr_mult),
            dict(lr_mult=2 * lr_mult)],
        'weight_filler': dict(type='xavier'),
        'bias_filler': dict(type='constant')
    }

    product_name = '{}{}{}'.format(product_prefix, out_layer, product_postfix)
    net[product_name] = L.InnerProduct(net[from_layer], num_output=num_output, **kwargs)
    if use_relu:
        relu_name = '{}/relu'.format(product_name)
        net[relu_name] = L.ReLU(net[product_name], in_place=True)


def lenet5_body(net, from_layer):
    conv_layer(net, from_layer, 'conv1', use_relu=False,
               num_output=20, kernel_size=5, pad=0, stride=1)
    net.pool1 = L.Pooling(net[net.keys()[-1]], kernel_size=2, stride=2, pool=P.Pooling.MAX)

    conv_layer(net, net.keys()[-1], 'conv2', use_relu=False,
               num_output=50, kernel_size=5, pad=0, stride=1)
    net.pool2 = L.Pooling(net[net.keys()[-1]], kernel_size=2, stride=2, pool=P.Pooling.MAX)

    inner_product_layer(net, net.keys()[-1], 'ip1', use_relu=True, num_output=500)
    inner_product_layer(net, net.keys()[-1], 'ip2', use_relu=False, num_output=10)


if __name__ == '__main__':
    n = caffe.NetSpec()
    n.data, n.label = L.Data(data_param=dict(batch_size=64, backend=P.Data.LMDB, source='path-to-lmdb'),
                             transform_param=dict(scale=1. / 255),
                             include=dict(phase=0),     # 0 for TRAIN and 1 for TEST
                             ntop=2)

    # todo -- how to make train Data layer and test Data layer share the same name
    n.data2, n.label2 = L.Data(data_param=dict(batch_size=100, backend=P.Data.LMDB, source='path-to-lmdb'),
                               transform_param=dict(scale=1. / 255),
                               include=dict(phase=1),
                               ntop=2)

    lenet5_body(n, from_layer='data')
    n.accuracy = L.Accuracy(n.ip2, n.label, include=dict(phase=1))
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)

    with open('/home/terry/github/my-caffemodel-scripts/example/lenet5_train.prototxt', 'w') as f:
        print('name: "{}_train_test"'.format('lenet5'), file=f)
        print(n.to_proto(), file=f)
