# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 23:03:55 2021

@author: Administrator
"""
import tensorflow as tf
import globalmap as GL

from typing import Tuple

def parse_exmp(current_decoder:str, serial_exmp: str, input_length: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    feature_dic = {
        'feature': tf.io.FixedLenFeature([input_length], tf.float32),
        'shape':  tf.io.VarLenFeature(tf.int64),
    }
    if current_decoder in ['Check-SF1','Check-SF2','Check-SF3']:
        feature_dic['label'] = tf.io.FixedLenFeature([input_length], tf.float32)
    elif current_decoder in ['SPA-1', 'NMS-1','QBP-SF1','QBP-SF2','QBP-SF3']:
        feature_dic['label'] = tf.io.FixedLenFeature([input_length], tf.int64)
    else:
        raise ValueError(f"Unknown decoder type: {current_decoder}")
    
    feats = tf.io.parse_single_example(serial_exmp, features=feature_dic)
    
    soft_input = feats['feature']
    label = feats['label']
    shape = tf.sparse.to_dense(feats['shape'], default_value=0)   
    return soft_input, label,shape

def get_dataset(current_decoder:str, fname: str, input_length: int) -> tf.data.Dataset:
    dataset = tf.data.TFRecordDataset(fname)
    
    @tf.autograph.experimental.do_not_convert
    def lambda_wrapper(x):
        return parse_exmp(current_decoder, x, input_length)
    
    dataset = dataset.map(
        lambda_wrapper,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    return dataset

def data_handler(current_decoder:str, input_length: int, file_name: str, batch_size: int = 1) -> tf.data.Dataset:
    dataset_train = get_dataset(current_decoder,file_name, input_length)
    dataset_train = dataset_train.batch(batch_size, drop_remainder=False)
    dataset_train = dataset_train.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset_train
  