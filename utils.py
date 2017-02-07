from __future__ import division,print_function
import math, os, json, sys, re
import cPickle as pickle
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
from operator import itemgetter, attrgetter, methodcaller
from collections import OrderedDict
import itertools
from itertools import chain

import pandas as pd
import PIL
from PIL import Image
#import cv2
from numpy.random import random, permutation, randn, normal, uniform, choice
from numpy import newaxis
import scipy
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import imread
from sklearn.metrics import confusion_matrix
#import bcolz
import h5py
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE

from IPython.lib.display import FileLink

import theano
from theano import shared, tensor as T
from theano.tensor.nnet import conv2d, nnet
from theano.tensor.signal import pool

import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.utils import np_utils
from keras.utils.io_utils import HDF5Matrix
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Reshape, merge, LSTM, Bidirectional
from keras.layers import TimeDistributed, Activation, SimpleRNN, GRU
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.regularizers import l2, activity_l2, l1, activity_l1
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils.layer_utils import layer_from_config
from keras.metrics import categorical_crossentropy, categorical_accuracy
from keras.layers.convolutional import *
#from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing import image, sequence
from keras.preprocessing.text import Tokenizer

from vgg16 import *
np.set_printoptions(precision=4, linewidth=100)


to_bw = np.array([0.299, 0.587, 0.114])
def gray(img):
    return np.rollaxis(img,0,3).dot(to_bw)
def to_plot(img):
    return np.rollaxis(img, 0, 3).astype(np.uint8)
def plot(img):
    plt.imshow(to_plot(img))


def floor(x):
    return int(math.floor(x))
def ceil(x):
    return int(math.ceil(x))

def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        if titles is not None:
            sp.set_title(titles[i], fontsize=18)
        plt.imshow(ims[i], interpolation=None if interp else 'none')


def do_clip(arr, mx):
    clipped = np.clip(arr, (1-mx)/1, mx)
    return clipped/clipped.sum(axis=1)[:, np.newaxis]


def get_batches(dirname, gen=image.ImageDataGenerator(), shuffle=True, batch_size=4, class_mode='categorical',
                target_size=(224,224)):
    return gen.flow_from_directory(dirname, target_size=target_size,
            class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)


def onehot(x):
    return to_categorical(x)


def wrap_config(layer):
    return {'class_name': layer.__class__.__name__, 'config': layer.get_config()}


def copy_layer(layer): return layer_from_config(wrap_config(layer))


def copy_layers(layers): return [copy_layer(layer) for layer in layers]


def copy_weights(from_layers, to_layers):
    for from_layer,to_layer in zip(from_layers, to_layers):
        to_layer.set_weights(from_layer.get_weights())


def copy_model(m):
    res = Sequential(copy_layers(m.layers))
    copy_weights(m.layers, res.layers)
    return res


def insert_layer(model, new_layer, index):
    res = Sequential()
    for i,layer in enumerate(model.layers):
        if i==index: res.add(new_layer)
        copied = layer_from_config(wrap_config(layer))
        res.add(copied)
        copied.set_weights(layer.get_weights())
    return res


def adjust_dropout(weights, prev_p, new_p):
    scal = (1-prev_p)/(1-new_p)
    return [o*scal for o in weights]


def get_data(path, target_size=(224,224)):
    batches = get_batches(path, shuffle=False, batch_size=1, class_mode=None, target_size=target_size)
    return np.concatenate([batches.next() for i in range(batches.nb_sample)])


def get_data_labels(path, target_size=(224,224)):
    batches = get_batches(path, shuffle=False, batch_size=1, class_mode='categorical', target_size=target_size)
    data, labels = zip(*[batches.next() for i in range(batches.nb_sample)])
    return np.squeeze(np.asarray(data)), np.squeeze(np.asarray(labels))


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    (This function is copied from the scikit docs.)
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()


def load_array(fname):
    return bcolz.open(fname)[:]

def load_array_partial(fname, start=0, end=1000):
    c=bcolz.open(fname)
    return c[start:end]

def append_array(arr, data):
    size = len(data)
    arr.resize(arr.shape[0]+size, axis=0)
    arr[-size:,] = data

def create_arrays(f, names, shapes):
    if(len(shapes)!=len(names)):
        raise ValueError('shapes and names should have same length. '
                         'Found: len(shapes) = %s, len(names) = %s' %
                         (len(shapes), len(names)))
    dsets = []
    for name, shape in zip(names, shapes):
        dsets.append(f.create_dataset(name, (0,)+shape[1:], maxshape=(None,)+shape[1:]))
    return tuple(dsets)

def get_batches(datagen):
    batch_size = datagen.batch_size
    for i in range(0, datagen.n, batch_size):
        yield datagen.next()

class FeatureSaver():
    def __init__(self, train_datagen, valid_datagen=None, test_datagen=None):
        self.train_datagen = train_datagen
        self.valid_datagen = valid_datagen
        self.test_datagen = test_datagen
        self.nb_classes = self.get_nb_classes()
    
    def get_nb_classes(self):
        class_name = self.train_datagen.__class__.__name__
        if(class_name=='DirectoryIterator'):
            return (self.train_datagen.nb_class,)
        else:
            return self.train_datagen.y.shape[1:]

    def run_epoch(self, datagen, model, feat_dset, label_dset=None):
        for tup in get_batches(datagen):
            features = model.predict_on_batch(tup[0])
            append_array(feat_dset, features)
            if((len(tup)>1) and (label_dset!=None)):
                append_array(label_dset, tup[1])

    def save_train(self, model, f, num_epochs=10):
        datagen = self.train_datagen

        data_shape = model.layers[-1].output_shape
        label_shape = (None,)+self.nb_classes
        feat_dset, label_dset = create_arrays(f, ['train_features', 'train_labels'],
                                              [data_shape, label_shape])

        for i in range(num_epochs):
            self.run_epoch(datagen, model, feat_dset, label_dset)

    def save_valid(self, model, f):
        datagen = self.valid_datagen

        data_shape = model.layers[-1].output_shape
        label_shape = (None,)+self.nb_classes
        feat_dset, label_dset = create_arrays(f, ['valid_features', 'valid_labels'],
                                              [data_shape, label_shape])
        self.run_epoch(datagen, model, feat_dset, label_dset)

    def save_test(self, model, f):
        datagen = self.test_datagen

        data_shape = model.layers[-1].output_shape
        label_shape = (None,)+self.nb_classes

        feat_dset, label_dset = create_arrays(f, ['test_features', 'test_labels'],
                                              [data_shape, label_shape])
        self.run_epoch(datagen, model, feat_dset)

class DataSaver():
    def __init__(self, path_folder, gen=image.ImageDataGenerator(), 
                batch_size=64, target_size=(224,224)):
        self.path = path_folder
        self.gen = gen
        self.results_path = self.path+'results/'

        self.batch_size = batch_size
        self.target_size = target_size

    def save_images(self, f, split_names=['train', 'valid']):
        for split_name in split_names:
            if(split_name=='train'):
                gen = self.gen
            else:
                gen = image.ImageDataGenerator()
            path_name = self.path+split_name+'/'
            datagen = gen.flow_from_directory(path_name, 
                                              target_size=self.target_size,
                                              batch_size=self.batch_size, 
                                              shuffle=False)
            data_shape = (None,)+datagen.image_shape
            label_shape = (None,)+(datagen.nb_class,)

            data_dset, label_dset = create_arrays(f, [split_name+'_data', split_name+'_labels'], 
                                                  [data_shape, label_shape])
            for data, labels in get_batches(datagen):
                append_array(data_dset, data)
                if(split_name!='test'):
                    append_array(label_dset, labels)

    def save_train(self, fname='dataset.h5'):
        f = h5py.File(self.results_path+fname, 'w')
        self.save_images(f, split_names=['train'])

    def save_trainval(self, fname='dataset.h5'):
        f = h5py.File(self.results_path+fname, 'w')
        self.save_images(f, split_names=['train', 'valid'])

    def save_all(self, fname='dataset.h5'):
        f = h5py.File(self.results_path+fname, 'w')
        self.save_images(f, split_names=['train', 'valid', 'test'])

def mk_size(img, r2c):
    r,c,_ = img.shape
    curr_r2c = r/c
    new_r, new_c = r,c
    if r2c>curr_r2c:
        new_r = floor(c*r2c)
    else:
        new_c = floor(r/r2c)
    arr = np.zeros((new_r, new_c, 3), dtype=np.float32)
    r2=(new_r-r)//2
    c2=(new_c-c)//2
    arr[floor(r2):floor(r2)+r,floor(c2):floor(c2)+c] = img
    return arr


def mk_square(img):
    x,y,_ = img.shape
    maxs = max(img.shape[:2])
    y2=(maxs-y)//2
    x2=(maxs-x)//2
    arr = np.zeros((maxs,maxs,3), dtype=np.float32)
    arr[floor(x2):floor(x2)+x,floor(y2):floor(y2)+y] = img
    return arr


def vgg_ft(out_dim):
    vgg = Vgg16()
    vgg.ft(out_dim)
    model = vgg.model
    return model


def get_classes(path):
    batches = get_batches(path+'train', shuffle=False, batch_size=1)
    val_batches = get_batches(path+'valid', shuffle=False, batch_size=1)
    test_batches = get_batches(path+'test', shuffle=False, batch_size=1)
    return (val_batches.classes, batches.classes, onehot(val_batches.classes), onehot(batches.classes),
        val_batches.filenames, batches.filenames, test_batches.filenames)


def split_at(model, layer_type):
    layers = model.layers
    layer_idx = [index for index,layer in enumerate(layers)
                 if type(layer) is layer_type][-1]
    return layers[:layer_idx+1], layers[layer_idx+1:]


class MixIterator(object):
    def __init__(self, iters):
        self.iters = iters
        self.multi = type(iters) is list
        if self.multi:
            self.N = sum([it[0].N for it in self.iters])
        else:
            self.N = sum([it.N for it in self.iters])

    def reset(self):
        for it in self.iters: it.reset()

    def __iter__(self):
        return self

    def next(self, *args, **kwargs):
        if self.multi:
            nexts = [[next(it) for it in o] for o in self.iters]
            n0s = np.concatenate([n[0] for n in o])
            n1s = np.concatenate([n[1] for n in o])
            return (n0, n1)
        else:
            nexts = [next(it) for it in self.iters]
            n0 = np.concatenate([n[0] for n in nexts])
            n1 = np.concatenate([n[1] for n in nexts])
            return (n0, n1)

def resize_image(img, scale=0.5):
    if((img.ndim==3) and (img.shape[0]<=3)):
        img = np.transpose(img, (1,2,0))
        img = misc.imresize(img, scale)
        img = np.transpose(img, (2,0,1))
    else:
        img = misc.imresize(img, 0.5)
    return img
