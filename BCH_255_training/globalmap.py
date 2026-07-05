"""
Created on Thu Nov 11 23:58:09 2021

@author: Administrator
"""# dictionary operations including adding,deleting or retrieving
import os,sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import read_TFdata as Reading
import fill_matrix_info as Fill_matrix

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

#global parameters setting
def global_setting(argv):
    #command line arguments
    set_map('snr_lo', float(argv[1]))
    set_map('snr_hi', float(argv[2]))
    set_map('snr_num', int(sys.argv[3]))
    set_map('unit_batch_size', int(argv[4]))
    set_map('num_batch_train', int(argv[5]))
    set_map('num_iterations', int(argv[6]))
    set_map('H_filename', argv[7])
    set_map('selected_decoder_type', argv[8])
    
    # the training/testing paramters setting for selected_decoder_type
    set_map('loss_process_indicator', True)
    set_map('ALL_ZEROS_CODEWORD_TRAINING', False)
    set_map('loss_coefficient',5)
    
    set_map('epochs',100)
    set_map('initial_learning_rate', 0.001)
    set_map('decay_rate', 0.99)
    set_map('decay_step', 200)
    set_map('iterate_termination_step',500000)
    set_map('termination_threshold',100)

    set_map('reduction_iteration',4)        
    set_map('redundancy_factor',2)
    set_map('num_shifts',3)
    
    #set_map('group_size',5)
    
    set_map('print_interval',100)

    set_map('regular_matrix',False)
    set_map('generate_extended_parity_check_matrix',True)
    
    set_map('training_model_phase',False)
    set_map('collect_failure_phase',True)
    set_map('global_training',True)
    set_map('merge_retrain_data',False)
    set_map('max_iteration_check_data',10)
    
    set_map('prefix_str','bch')
    
    #filling parity check matrix info
    H_filename = get_map('H_filename')
    code = Fill_matrix.Code(H_filename)
    #store it onto global space
    set_map('code_parameters', code)


    set_map('generate_check_data',False)
    
    
    set_map('clip_max',5.)
    

    snr_lo = get_map('snr_lo')  
    snr_hi = get_map('snr_hi')  
    snr_num = get_map('snr_num')
    SNRs = np.round( np.linspace(snr_lo,snr_hi,snr_num), 1)
    decoder_type = get_map('selected_decoder_type')
    if decoder_type in ['SPA-1','Check-SF1','Check-SF2','Check-SF3','QBP-SF1','QBP-SF2','QBP-SF3']:
        sigma_square_list = []
        for snr in list(SNRs):
            sigma_square = 1. / (2 * (float(code.k)/float(code.n)) * 10**(snr/10))
            sigma_square_list.append(sigma_square)
        set_map('noise_variance_list', sigma_square_list)
        default_normalizor_dict = {3.5:-1.8,4.0:-2.0,4.5:-2.2,5.0:-2.4,5.5:-2.6,6.0:-2.8,6.5:-3.0}
    if decoder_type == 'NMS-1':
        default_normalizor_dict = {3.5:-3.0,4.0:-2.9,4.5:-2.8,5.0:-2.7,5.5:-2.6,6.0:-2.5}
    set_map('default_normalizor_dict', default_normalizor_dict)
    return SNRs
        
def logistic_setting(current_decoder,snr=''):
    prefix_str = get_map('prefix_str')
    n_iteration = get_map('num_iterations')
    num_shifts = get_map('num_shifts') 
    snr_lo = get_map('snr_lo')
    snr_hi = get_map('snr_hi')  
    if snr:
        snr_info = f'{snr_lo}-{snr_hi}dB/{snr}dB/'
    else:
        snr_info = f'{snr_lo}-{snr_hi}dB/'
    basic_dir = f'./ckpts/{snr_info}{current_decoder}/{n_iteration}th/'
    reduction_iteration = get_map('reduction_iteration')
    redundancy_factor = get_map('redundancy_factor')   
    ckpt_nm = f'{prefix_str}-ckpt'
    intermediate_dir = basic_dir+f'IF{reduction_iteration}-{redundancy_factor}-Ns-{num_shifts}/'
    ckpts_dir = intermediate_dir+ckpt_nm    
    restore_step = ''
    restore_info = [ckpts_dir,ckpt_nm,restore_step]
    return restore_info

def base_dataset(current_decoder,code, unit_batch_size,snr=None):
    prefix_str = get_map('prefix_str')
    code_length = code.n
    snr_lo = round(get_map('snr_lo'), 2)
    snr_hi = round(get_map('snr_hi'), 2)
    if snr:
        data_dir = f'../Training_data_gen_{code_length}/data/snr{snr_lo}-{snr_hi}dB/{snr}dB/'
    else:
        data_dir = f'../Training_data_gen_{code_length}/data/snr{snr_lo}-{snr_hi}dB/'        
    file_name = f'{prefix_str}-train-allzero.tfrecord' if get_map('ALL_ZEROS_CODEWORD_TRAINING') else f'{prefix_str}-train-nonzero.tfrecord'
    file_path = data_dir + file_name
    return Reading.data_handler(current_decoder,code_length, file_path, unit_batch_size).cache()  # no cache yet

def build_training_dataset(current_decoder,code, unit_batch_size,snr=None):
    dataset = base_dataset(current_decoder,code, unit_batch_size,snr)
    #return dataset.shuffle(1000,seed=40).repeat().prefetch(tf.data.AUTOTUNE)
    return dataset.shuffle(1000).repeat().prefetch(tf.data.AUTOTUNE)

def data_setting(code,unit_batch_size,snr=''):
    prefix_str = get_map('prefix_str')
    #training data directory
    code_length = code.n
    snr_lo = round(get_map('snr_lo'),2)
    snr_hi = round(get_map('snr_hi'),2)
    n_iteration = get_map('num_iterations')
    basic_dir = f'../Training_data_gen_{code_length}/data/snr{snr_lo}-{snr_hi}dB/'
    decoder_type = get_map('selected_decoder_type') 
    if decoder_type in ['Check-SF1','Check-SF2','Check-SF3']:
        decoder_str = 'Check-SF'
    else:
        decoder_str = decoder_type
    if snr:
        data_dir = basic_dir+f'{snr}dB/'
        gen_data_dir = f'{data_dir}{decoder_str}/{n_iteration}th/'
        # reading in training/validating data;make dataset iterator
        if get_map('ALL_ZEROS_CODEWORD_TRAINING'):
            spec_str ='-allzero'
        else:
            spec_str = '-nonzero'
        file_name = f'{prefix_str}-train{spec_str}.tfrecord'
        path_file = data_dir+file_name
        #preparing batch iterator of data file        
        if decoder_type in ['Check-SF1','Check-SF2','Check-SF3'] and get_map('generate_check_data'):
            dataset_train = Reading.data_handler('SPA-1',code_length,path_file,unit_batch_size)
            dataset_train = dataset_train.take(10)
        else:
            dataset_train = Reading.data_handler(decoder_type,code_length,path_file,unit_batch_size)
        #selected_ds = dataset_train.shuffle(1000,seed=40).cache()
        selected_ds = dataset_train.shuffle(1000).cache()
    else:
        gen_data_dir = basic_dir
        selected_ds = ''
        
    if not os.path.exists(gen_data_dir):
      os.makedirs(gen_data_dir)
      
    return gen_data_dir,selected_ds

def reading_approx_training_data(current_decoder):
    code = get_map('code_parameters')
    unit_batch_size = get_map('unit_batch_size')
    #training data directory
    code_length = code.H.shape[1]
    if current_decoder in ['Check-SF1','Check-SF2','Check-SF3']:
        input_length = tf.reduce_sum((code.H)[0])        
    decoder_str = 'Check-SF'
    snr_lo = round(get_map('snr_lo'),2)
    snr_hi = round(get_map('snr_hi'),2)
    n_iteration = get_map('num_iterations')
    if get_map('ALL_ZEROS_CODEWORD_TRAINING'):  
        distinct_string = 'allzero'
    else:
        distinct_string = 'nonzero'

    data_dir = f'../Training_data_gen_{code_length}/data/snr{snr_lo}-{snr_hi}dB/{decoder_str}/{n_iteration}th/'        
    input_file_name = f'merged_train-{distinct_string}.tfrecord'

    # reading in training/validating data;make dataset iterator
    input_dir_file = data_dir+input_file_name
    if not os.path.exists(input_dir_file):
        print(f"File {input_dir_file} No existence! Generate training_data before starting off training!")
        sys.exit(1)  # 
    input_dataset_train = Reading.data_handler(current_decoder,input_length,input_dir_file,unit_batch_size)
    #preparing batch iterator of data file
    selected_ds = (input_dataset_train
               .cache()
               #.shuffle(1000,seed=40)
               .shuffle(1000)
               .repeat()
               .prefetch(tf.data.AUTOTUNE))
    return selected_ds


def optimizer_setting():
    #optimizing settings
    decay_rate = get_map('decay_rate')
    initial_learning_rate = get_map('initial_learning_rate')
    decay_steps = get_map('decay_step')
    exponential_decay = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps, decay_rate,staircase=True)
    return exponential_decay

def log_setting(restore_info,checkpoint):
    n_iteration = get_map('num_iterations')
    decoder_type = get_map('selected_decoder_type')
    (ckpts_dir,ckpt_nm,_) = restore_info
    # summary recorder
    tensorboard_dir = f'./tensorboard/{decoder_type}/{n_iteration}th'
    summary_writer = tf.summary.create_file_writer(tensorboard_dir)     # the parameter is the log folder we created
    manager_current = tf.train.CheckpointManager(checkpoint, directory=ckpts_dir, checkpoint_name=ckpt_nm, max_to_keep=5)
    logger_info = (summary_writer,manager_current)
    return logger_info

