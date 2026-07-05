# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 20:18:23 2022

@author: lgw
"""
import numpy as np
import math,os
from scipy import integrate
import globalmap as GL
import tensorflow as tf

#np.random.seed(0)

def f1(x,mid_sigma): 
  y=2/(x**2)*f_w(x,mid_sigma)
  return y
def f2(x,mid_sigma):
  y=4*(1/(x**2)+1/(x**4))*f_w(x,mid_sigma)
  return y
def f_w(x,mid_sigma):
  t = abs(x-mid_sigma)
  y= math.exp(-t)
  return y
#writing all data to tfrecord file
def make_tfrecord(data, out_filename):
  feats,labels = data
  ndatas = len(labels)
  os.makedirs(os.path.dirname(out_filename), exist_ok=True)
  with tf.io.TFRecordWriter(out_filename) as file_writer:
      for inx in range(ndatas):   
          feature,label = feats[inx], labels[inx]
          feat_shape = feats[inx].shape
          record_bytes = tf.train.Example(features=tf.train.Features(feature={
              "feature": tf.train.Feature(float_list=tf.train.FloatList(value=feature)),
              "label": tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
              "shape":  tf.train.Feature(int64_list=tf.train.Int64List(value=list(feat_shape)))
          })).SerializeToString()    
          file_writer.write(record_bytes)    

def mixting_even_samples(code,SNRs,max_frame):
    n = code.check_matrix_column
    k = code.k 
    noise = np.random.randn(max_frame,n)
    #starting off the data generating process
    SNR1 = SNRs[0]
    SNR2 = SNRs[1]    
    sigma1 =  np.sqrt(1. / (2 * (float(k)/float(n)) * 10**(SNR1/10)))
    sigma2 =  np.sqrt(1. / (2 * (float(k)/float(n)) * 10**(SNR2/10)))
    mid_SNR = (SNR1+SNR2)/2
    mid_sigma = np.sqrt(1. / (2 * (float(k)/float(n)) * 10**(mid_SNR/10)))
    #weight_coefficient for valid density
    if SNR1!=SNR2:
        tmp,_ = integrate.quad(f_w,sigma1,sigma2,args=(mid_sigma))
        weight_coefficient = 1/tmp
        tmp_mean,_ = integrate.quad(f1, sigma1,sigma2,args=(mid_sigma))
        new_mean = weight_coefficient*tmp_mean
        tmp_variance,_ = integrate.quad(f2, sigma1,sigma2,args=(mid_sigma))
        new_variance  = weight_coefficient*tmp_variance- new_mean**2
        #print(new_mean,new_variance)
        sigma = np.sqrt(new_variance)
    else:
        sigma = sigma1
        new_mean = 1
    noise *= sigma 
    noise +=new_mean
    # generate random codewords
    if GL.get_map('ALL_ZEROS_CODEWORD_TRAINING'):
        training_data = noise
    else:
        rand_message = np.random.randint(0,2,size=[max_frame,k],dtype=int)
        codewords = rand_message.dot(code.G)%2
        training_data = np.where(codewords==0,noise,-noise)
        training_data_labels = codewords
    return [training_data],[training_data_labels]    
  
def training_data_generating(code,SNRs,max_frame):
    #em_estimator = BPSK_EM_Estimator()
    #retrieving global paramters of the code
    n = code.check_matrix_column
    k = code.k 
    training_data_labels = np.zeros((max_frame,n),dtype=np.int64)
    #starting off the data generating process
    mix_samples_indicator = GL.get_map('mix_samples_indicator')
    if mix_samples_indicator:
        train_data_list,train_label_list = mixting_even_samples(code,SNRs,max_frame)
    else:
        train_data_list = []
        train_label_list = []
        for snr in SNRs:    
            noise = np.random.randn(max_frame,n)     
            sigma =  np.sqrt(1. / (2 * (float(k)/float(n)) * 10**(snr/10)))
            new_mean = 1
            noise *= sigma 
            noise +=new_mean
            # generate random codewords
            if GL.get_map('ALL_ZEROS_CODEWORD_TRAINING'):
                training_data = noise
                training_labels = training_data_labels
            else:
                rand_message = np.random.randint(0,2,size=[max_frame,k],dtype=int)
                codewords = rand_message.dot(code.G)%2
                training_data = np.where(codewords==0,noise,-noise)
                training_labels = codewords
            train_data_list.append(training_data)
            train_label_list.append(training_labels)
    return train_data_list,train_label_list

# class BPSK_EM_Estimator:
#     def __init__(self, max_iter=100, tol=1e-6):
#         self.max_iter = max_iter
#         self.tol = tol        
#     def gaussian_pdf(self, x, mu, sigma2):
#         return tf.exp(-(x - mu)**2 / (2 * sigma2)) / tf.sqrt(2 * np.pi * sigma2)
    
#     def fit(self, soft_inputs, verbose=True):
#         y = tf.reshape(soft_inputs, [-1])
#         n = tf.cast(tf.shape(y)[0], tf.float32)
#         y_sorted = tf.sort(y)
#         split_idx = tf.cast(n/2, tf.int32)        
#         mu_neg_init = tf.reduce_mean(y_sorted[:split_idx])   # 1 -> -1
#         mu_pos_init = tf.reduce_mean(y_sorted[split_idx:])   # 0 -> +1   
#         sigma_neg2_init = tf.reduce_mean((y_sorted[:split_idx] - mu_neg_init)**2)
#         sigma_pos2_init = tf.reduce_mean((y_sorted[split_idx:] - mu_pos_init)**2)
#         sigma2_init = (sigma_neg2_init + sigma_pos2_init) / 2
#         if mu_neg_init > mu_pos_init:
#             mu_neg_init, mu_pos_init = mu_pos_init, mu_neg_init
#         mu_neg = tf.Variable(mu_neg_init, dtype=tf.float32)
#         mu_pos = tf.Variable(mu_pos_init, dtype=tf.float32)
#         sigma2 = tf.Variable(sigma2_init, dtype=tf.float32)   
#         pi = 0.5  # prior probability 
#         prev_log_likelihood = -np.inf
        
#         for iteration in range(self.max_iter):
#             # E-Step: posterior probability calculation
#             # prob_bit0: 
#             prob_bit0 = pi * self.gaussian_pdf(y, mu_pos, sigma2)
#             # prob_bit1: 
#             prob_bit1 = (1-pi) * self.gaussian_pdf(y, mu_neg, sigma2)
#             gamma_bit0 = prob_bit0 / (prob_bit0 + prob_bit1 + 1e-10)
#             gamma_bit1 = 1 - gamma_bit0
#             log_likelihood = tf.reduce_sum(tf.math.log(prob_bit0 + prob_bit1 + 1e-10))
#             # M-Step: udpate parameters
#             n_bit0 = tf.reduce_sum(gamma_bit0)
#             n_bit1 = tf.reduce_sum(gamma_bit1)         
#             # update means
#             new_mu_pos = tf.reduce_sum(gamma_bit0 * y) / (n_bit0 + 1e-10)   # 比特0的均值
#             new_mu_neg = tf.reduce_sum(gamma_bit1 * y) / (n_bit1 + 1e-10)   # 比特1的均值
            
#             # udpate the sole variance
#             new_sigma2 = (tf.reduce_sum(gamma_bit0 * (y - new_mu_pos)**2) + 
#                          tf.reduce_sum(gamma_bit1 * (y - new_mu_neg)**2)) / n            
#             # apply updates
#             mu_pos.assign(new_mu_pos)
#             mu_neg.assign(new_mu_neg)
#             sigma2.assign(new_sigma2)           
#             # check convergence
#             if iteration > 0:
#                 if tf.abs(log_likelihood - prev_log_likelihood) < self.tol:
#                     if verbose:
#                         print(f"converging at {iteration}-th iteration")
#                     break           
#             prev_log_likelihood = log_likelihood           
#             if verbose and iteration % 10 == 0:
#                 print(f"Iter {iteration}: mu_neg={mu_neg.numpy():.4f} (bit 1), "
#                       f"mu_pos={mu_pos.numpy():.4f} (bit 0), "
#                       f"sigma2={sigma2.numpy():.4f}, logL={log_likelihood.numpy():.2f}")       
#         return {
#             'mu_bit0': mu_pos.numpy(),      # 0 -> +1
#             'mu_bit1': mu_neg.numpy(),      # 1 -> -1
#             'sigma2': sigma2.numpy()
#         }
