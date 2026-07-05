# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 20:18:23 2022

@author: lgw
"""
import tensorflow as tf
import globalmap as GL

#np.random.seed(0)


def get_tfrecords_example(feature, label):
    selected_decoder = GL.get_map('selected_decoder_type')
    tfrecords_features = {}
    feat_shape = feature.shape
    if selected_decoder in ['Check-SF1','Check-SF2','Check-SF3']:
        tfrecords_features['label'] = tf.train.Feature(float_list=tf.train.FloatList(value=label))
    if selected_decoder in ['SPA-1', 'NMS-1','QBP-SF1','QBP-SF2','QBP-SF3']:
        tfrecords_features['label'] = tf.train.Feature(int64_list=tf.train.Int64List(value=label))
    tfrecords_features['feature'] = tf.train.Feature(float_list=tf.train.FloatList(value=feature))
    tfrecords_features['shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(feat_shape)))
    return tf.train.Example(features = tf.train.Features(feature = tfrecords_features))
#writing all data to tfrecord file
def make_tfrecord(data, out_filename):
    feats,labels = data
    tfrecord_wrt = tf.io.TFRecordWriter(out_filename)
    ndatas = len(labels)
    for inx in range(ndatas):
        exmp = get_tfrecords_example(feats[inx].numpy(), labels[inx].numpy())
        exmp_serial = exmp.SerializeToString()
        tfrecord_wrt.write(exmp_serial)
    tfrecord_wrt.close()
 
