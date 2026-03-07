# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 23:58:09 2021

@author: Administrator
"""
import os
import tensorflow as tf
from tensorflow import keras
import fill_matrix_info as Fill_matrix
import read_TFdata as Reading
import ms_test as Decoder_module
# dictionary operations including adding,deleting or retrieving
map = {}
def set_map(key, value):
    map[key] = value
def del_map(key):
    try:
        del map[key]
    except KeyError :
        print ("key:'"+str(key)+"' non-existence")
def get_map(key):
    try:
        if key in "all":
            return map
        return map[key]
    except KeyError :
        print ("key:'"+str(key)+"' non-existence")

def global_setting(argv):
    #command line arguments
    set_map('snr_lo', float(argv[1]))
    set_map('snr_hi', float(argv[2]))
    set_map('snr_num', int(argv[3]))
    set_map('unit_batch_size', int(argv[4]))
    set_map('num_batch_train', int(argv[5]))
    set_map('num_iterations', int(argv[6]))
    set_map('H_filename', argv[7])
    set_map('selected_decoder_type', argv[8])
    
    # the training/testing paramters setting for selected_decoder_type
    set_map('loss_process_indicator', True)
    set_map('ALL_ZEROS_CODEWORD_TESTING', False)
    set_map('loss_coefficient',5)
    
    set_map('regular_matrix',False)
    set_map('generate_extended_parity_check_matrix',True)
    set_map('reduction_iteration',4)       
    set_map('redundancy_factor',2)
    set_map('num_shifts',5)
    
    set_map('print_interval',50)
    set_map('record_interval',50)

    set_map('decoding_threshold',100)
    set_map('Rayleigh_fading', False)
    set_map('reacquire_data',True)
    if get_map('Rayleigh_fading'):
        set_map('duration', 1)
        suffix = 'Rayleigh_awgn_duration_'+str(get_map('duration'))
    else:
        suffix = 'Awgn'
    set_map('suffix',suffix)
    set_map('enhanced_NMS_indicator',True)
    set_map('prefix_str','bch')
    #filling parity check matrix info
    H_filename = get_map('H_filename')
    code = Fill_matrix.Code(H_filename)
    #store it onto global space
    set_map('code_parameters', code)
    set_map('training_SNR',3.0)
    set_map('probe_MI',False)
    set_map('threshold_MI',0.9)
    set_map('validate_dataset_size',10)
    set_map('Drawing_EXIT_only',False)   
def logistic_setting():
    prefix_str = get_map('prefix_str')
    n_dims = get_map('code_parameters').n
    n_iteration = get_map('num_iterations')
    decoder_type = get_map('selected_decoder_type')
    num_shifts = get_map('num_shifts') 
    snr_lo = get_map('training_SNR')
    snr_hi = get_map('training_SNR')  
    snr_info = f'{snr_lo}-{snr_hi}dB/'
    basic_dir = f'../{prefix_str.upper()}_{n_dims}_training/ckpts/{snr_info}/{decoder_type}/{n_iteration}th/'
    reduction_iteration = get_map('reduction_iteration')
    redundancy_factor = get_map('redundancy_factor')   
    ckpt_nm = f'{prefix_str}-ckpt'
    intermediate_dir = basic_dir+f'IF{reduction_iteration}-{redundancy_factor}-ns-{num_shifts}/'
    ckpts_dir_par = intermediate_dir+'par/'
    ckpts_dir = intermediate_dir+ckpt_nm
    #create the directory if not existing
    if not os.path.exists(ckpts_dir_par):
        os.makedirs(ckpts_dir_par)      
    restore_step = ''
    restore_info = [ckpts_dir,ckpt_nm,ckpts_dir_par,restore_step]
    return restore_info

def log_setting():
    decoder_type = get_map('selected_decoder_type')
    n_iteration = get_map('num_iterations')
    logdir = './log/'
    if not os.path.exists(logdir):
        os.makedirs(logdir)   
    log_filename = logdir+'FER-'+decoder_type+'-'+str(n_iteration)+'th'+'.txt'
    return log_filename

def data_setting(snr):
    snr = round(snr,2)
    # reading in training/validating data;make dataset iterator
    if get_map('ALL_ZEROS_CODEWORD_TESTING'):
        ending = 'allzero'
    else:
        ending = 'nonzero'
    unit_batch_size = get_map('unit_batch_size')
    decoder_type = get_map('selected_decoder_type')
    n_iteration = get_map('num_iterations')
    snr_lo = round(get_map('snr_lo'),2)
    snr_hi = round(get_map('snr_hi'),2)
    code = get_map('code_parameters')
    suffix = get_map('suffix')
    n_dims = code.n
    data_dir = '../Testing_data_gen_'+str(n_dims)+'/data/snr'+str(snr_lo)+'-'+str(snr_hi)+'dB/'
    iput_file =  data_dir +'test-'+ending+str(snr)+'dB-'+suffix+'.tfrecord'
    output_dir = data_dir+'/'+str(decoder_type)+'/'+str(n_iteration)+'th/'+str(snr)+'dB/'
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    dataset_test = Reading.data_handler(n_dims,iput_file,unit_batch_size)
    #preparing batch iterator of data file
    #dataset_test = dataset_test.take(1000)
    return output_dir,dataset_test 
def retore_saved_model(restore_info,SE_instance):
    code = get_map('code_parameters')
    test_Model = Decoder_module.Decoding_model(SE_instance)
    # Explicitly build the model with dummy input
    dummy_input_shape = (None, code.n)  # Replace with actual shape
    test_Model.build(dummy_input_shape)  # ⚠️ This triggers build() in Decoder_Layer
    tf.print("Pre-restoration weight:", test_Model.trainable_variables)
    # save restoring info
    checkpoint = tf.train.Checkpoint(myAwesomeModel=test_Model)
    [ckpts_dir,ckpt_nm,ckpts_dir_par,restore_step] = restore_info 
    print("Ready to restore a saved latest or designated model!")
    ckpt = tf.train.get_checkpoint_state(ckpts_dir)
    if ckpt and ckpt.model_checkpoint_path: # ckpt.model_checkpoint_path means the latest ckpt
        if restore_step:
            if restore_step == 'latest':
                ckpt_f = tf.train.latest_checkpoint(ckpts_dir)
            else:
                ckpt_f = ckpts_dir+ckpt_nm+'-'+restore_step
            print('Loading wgt file: '+ ckpt_f)   
            status = checkpoint.restore(ckpt_f)
            tf.print("Post-restoration weight:", test_Model.trainable_variables)
            #status.assert_existing_objects_matched() 
            status.expect_partial()
    else:
        print('Error, no qualified file found')
    return test_Model

