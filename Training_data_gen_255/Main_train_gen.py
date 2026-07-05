# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 16:31:50 2021

@author: Administrator
"""
import numpy as np
import sys
import os
# Run as follows:
np.set_printoptions(precision=3)
import fill_matrix_info as Fill_matrix
import globalmap as GL
import data_generating as Data_gen
# python main.py 1.5 3 16 1e4   wimax_1056_0.83.alist            
#command line arguments
sys.argv = "python 3.5 6.0 6 100 1000 BCH_255_239_strip.alist".split()
GL.set_map('snr_lo', float(sys.argv[1]))
GL.set_map('snr_hi', float(sys.argv[2]))
GL.set_map('snr_num', int(sys.argv[3]))
GL.set_map('batch_size', int(sys.argv[4]))
GL.set_map('training_batch_number', int(sys.argv[5]))
GL.set_map('H_filename', sys.argv[6])

# setting global parameters
H_filename=GL.get_map('H_filename')
batch_size = GL.get_map('batch_size')

code = Fill_matrix.Code(H_filename)
GL.set_map('code_parameters', code)
GL.set_map('ALL_ZEROS_CODEWORD_TRAINING', False)
GL.set_map('extended_input', False)
GL.set_map('prefix_str', 'bch')
GL.set_map('mix_samples_indicator', False)

#training setting
#retrieving global paramters of the code
n = code.check_matrix_column
train_batch = GL.get_map('training_batch_number')

snr_lo = GL.get_map('snr_lo')
snr_hi = GL.get_map('snr_hi')
snr_num = GL.get_map('snr_num')
SNRs = np.linspace(snr_lo,snr_hi, snr_num)

#create directory if not existence    
snr_info = f'snr{snr_lo:.1f}-{snr_hi:.1f}dB'  
basic_dir = f'./data/{snr_info}'
if not os.path.exists(basic_dir):
  os.makedirs(basic_dir) 
nDatas_train = train_batch*batch_size  
#generating training data
train_data_list,train_labels_list = Data_gen.training_data_generating(code,SNRs,nDatas_train)   
# make training set file
prefix_str = GL.get_map('prefix_str')
suffix_str = ''
if GL.get_map('extended_input'):
    suffix_str = '-extended'
if GL.get_map('ALL_ZEROS_CODEWORD_TRAINING'): 
    spe_str = '-allzero'
else:
    spe_str = '-nonzero'
for i,snr in enumerate(SNRs):
    data_file_dir = f'{basic_dir}/{snr:.1f}dB'
    path_file = f'{data_file_dir}/{prefix_str}-train{spe_str}{suffix_str}.tfrecord'
    data = (train_data_list[i],train_labels_list[i])
    Data_gen.make_tfrecord(data,out_filename=path_file)
print("Data for training generated successfully!")