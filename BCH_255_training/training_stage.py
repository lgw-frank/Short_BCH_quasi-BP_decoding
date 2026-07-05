# -*- coding: utf-8 -*-
import numpy as np
np.set_printoptions(precision=3)
import tensorflow  as tf
import globalmap as GL
import ms_decoder_dense as Decoder_module
import re
import data_generating as Data_gen

def fetch_model_info(restore_info):
    [ckpts_dir,ckpt_nm,restore_step] = restore_info 
    start_step = 0
    ckpt_f = ''
    print("Ready to restore a saved latest or designated model!")
    ckpt = tf.train.get_checkpoint_state(ckpts_dir)
    if not ckpt:
        ckpts_dir = re.sub(r'/\d*\.?\d+dB/', '/', ckpts_dir, count=1)
        ckpt = tf.train.get_checkpoint_state(ckpts_dir)
    if ckpt and ckpt.model_checkpoint_path: # ckpt.model_checkpoint_path means the latest ckpt
      if restore_step == 'latest':
        ckpt_f = tf.train.latest_checkpoint(ckpts_dir)
        start_step = int(ckpt_f.split('-')[-1]) + 1
      elif int(restore_step or '0') > 0:
        ckpt_f = ckpts_dir+ckpt_nm+'-'+restore_step
        start_step = int(restore_step)+1
      print('Loading wgt file: '+ ckpt_f)   
      if not restore_step:
          print('Relaunching training from scratch NOW')
    else:
      print('Model file no found, start from scratch NOW!')
    return start_step,ckpt_f


def restore_simple_model(current_decoder,restore_info):
    [ckpts_dir,_,restore_step] = restore_info 
    match = re.search(r'/(\d+(?:\.\d+)?)dB/', ckpts_dir)
    if match:
        snr = float(match.group(1))   
        default_normalizor_dict = GL.get_map('default_normalizor_dict') 
        default_initial = default_normalizor_dict.get(snr)
    else:
        default_initial = -2.0
    unit_batch_size = GL.get_map('unit_batch_size')
    code = GL.get_map('code_parameters')
    exponential_decay = GL.optimizer_setting()
    clip_max = GL.get_map('clip_max')
    
    if current_decoder in ['Check-SF1','Check-SF2','Check-SF3'] :
        optimizer =  tf.keras.optimizers.Adam(exponential_decay,clipnorm=clip_max)
        
        if current_decoder == 'Check-SF1':
            input_width = np.sum(code.H[0])
            NN_model = Decoder_module.conv_sf1_bitwise(input_width)
        if current_decoder == 'Check-SF2':  
            input_width = np.sum(code.H[0])        
            NN_model = Decoder_module.conv_sf2_bitwise(input_width)
        if current_decoder == 'Check-SF3': 
            input_width = np.sum(code.H[0])-1          
            NN_model = Decoder_module.conv_sf3_bitwise(input_width)
        train_iterator = GL.reading_approx_training_data(current_decoder) 
        # Explicitly build the model with dummy input
        dummy_input_shape = (None, input_width,1)  # Replace with actual shape
        NN_model.build(dummy_input_shape)  # ⚠️ This triggers build() in Decoder_Layer       
        tf.print(f'\nPre-restoration weight:{NN_model.trainable_variables[-1].numpy():.3f}')
    
    if current_decoder == 'SPA-1':
        optimizer =  tf.keras.optimizers.Adam(exponential_decay,clipnorm=clip_max)
        if GL.get_map('generate_check_data'):
            NN_model = Decoder_module.SPAE_model(initial=default_initial)  
        else:
            NN_model = Decoder_module.SPA_model(initial=default_initial)  
        # Explicitly build the model with dummy input
        dummy_input_shape = (None, code.n)  # Replace with actual shape
        NN_model.build(dummy_input_shape)  # ⚠️ This triggers build() in Decoder_Layer
        tf.print(f'\nPre-restoration weight:{NN_model.trainable_variables[-1].numpy():.3f}')
        train_iterator = GL.build_training_dataset(current_decoder,code,unit_batch_size,snr)
    
    if current_decoder == 'NMS-1':
        optimizer =  tf.keras.optimizers.Adam(exponential_decay,clipnorm=clip_max)
        NN_model = Decoder_module.NMS_model(initial = default_initial)          
        # Explicitly build the model with dummy input
        dummy_input_shape = (None, code.n)  # Replace with actual shape
        NN_model.build(dummy_input_shape)  # ⚠️ This triggers build() in Decoder_Layer
        tf.print(f'\nPre-restoration weight:{NN_model.trainable_variables[-1].numpy():.3f}')
        train_iterator = GL.build_training_dataset(current_decoder,code,unit_batch_size,snr)
    
    # save restoring info
    checkpoint = tf.train.Checkpoint(model=NN_model, optimizer=optimizer)
    logger_info = GL.log_setting(restore_info,checkpoint)
    #unpack related info for restoraging
    start_step = 0
    ckpt_f = ''
    #unpack related info for restoraging
    [_,_,restore_step] = restore_info 
    if  restore_step !=  '':
        start_step,ckpt_f = fetch_model_info(restore_info)
        if ckpt_f:
            status = checkpoint.restore(ckpt_f)
            status.expect_partial()
            tf.print(f'\nPost-restoration weight:{NN_model.trainable_variables[-1].numpy():.3f}')
    train_info = [start_step,exponential_decay, optimizer,train_iterator,\
                  logger_info]
    return NN_model,train_info

def restore_saved_model(current_decoder,restore_info):
    if current_decoder in ['NMS-1','SPA-1','Check-SF1','Check-SF2','Check-SF3']:
        NN_model,train_info = restore_simple_model(current_decoder,restore_info)
    if  current_decoder in ['QBP-SF1','QBP-SF2','QBP-SF3']:
        NN_model,train_info = restore_complex_model(current_decoder,restore_info)
    return NN_model,train_info

def restore_complex_model(current_decoder,restore_info):   
    [ckpts_dir,_,restore_step] = restore_info 
    match = re.search(r'/(\d+(?:\.\d+)?)dB/', ckpts_dir)
    if match:
        snr = float(match.group(1))   
    default_normalizor_dict = GL.get_map('default_normalizor_dict') 
    if snr:
        default_initial = default_normalizor_dict.get(snr)
    else:
        default_initial = -2.0
    unit_batch_size = GL.get_map('unit_batch_size')
    code = GL.get_map('code_parameters')
    clip_max = GL.get_map('clip_max')
    exponential_decay  = GL.optimizer_setting()
    optimizer =  tf.keras.optimizers.Adam(exponential_decay,clipnorm=clip_max)
    if current_decoder in ['QBP-SF1','QBP-SF2','QBP-SF3']:
        if current_decoder == 'QBP-SF1':
            input_width = np.sum(code.H[0])
            check_model = Decoder_module.conv_sf1_bitwise(input_width)  
        if current_decoder == 'QBP-SF2':
            input_width = np.sum(code.H[0])
            check_model = Decoder_module.conv_sf2_bitwise(input_width)      
        if current_decoder == 'QBP-SF3':
            input_width = np.sum(code.H[0])-1
            check_model = Decoder_module.conv_sf3_bitwise(input_width-1)
        if GL.get_map('generate_check_data'):
            NN_model = Decoder_module.SPAE_model(check_model=check_model)  
        else:         
            NN_model = Decoder_module.SPA_model(check_model=check_model,initial=default_initial)  
        # Explicitly build the model with dummy input
        dummy_input_shape = (None, code.n)  # Replace with actual shape
        NN_model.build(dummy_input_shape)  # ⚠️ This triggers build() in Decoder_Layer
        tf.print(f'\nPre-restoration weight:{NN_model.trainable_variables[-1].numpy():.3f}')
        if GL.get_map('global_training'):
            train_iterator = GL.build_training_dataset(current_decoder,code,unit_batch_size)
        else:
            train_iterator = GL.build_training_dataset(current_decoder,code,unit_batch_size,snr)
    # save restoring info
    checkpoint = tf.train.Checkpoint(model=NN_model, optimizer = optimizer)
    logger_info = GL.log_setting(restore_info,checkpoint)
    #unpack related info for restoraging
    start_step = 0
    ckpt_f = ''
    #unpack related info for restoraging
    [_,_,restore_step] = restore_info 
    if  restore_step !=  '':
        start_step,ckpt_f = fetch_model_info(restore_info)
        if ckpt_f:
            status = checkpoint.restore(ckpt_f)
            status.expect_partial()
            #status.assert_consumed()
            tf.print(f'\nPost-restoration weight:{NN_model.trainable_variables[-1].numpy():.3f}')
        else:
            start_step = 0
            NN_model = restore_QBP_SSF(current_decoder,NN_model,snr)
    train_info = [start_step,exponential_decay,optimizer,train_iterator,logger_info]
    if  (restore_step ==  '') and current_decoder in ['QBP-SF1','QBP-SF2','QBP-SF3']:
        NN_model = restore_QBP_SSF(current_decoder,NN_model,snr)
    return NN_model, train_info

def restore_QBP_SSF(selected_decoder,NN_model,snr=''):
    if selected_decoder == 'QBP-SF1':  
        current_decoder = 'Check-SF1'
    if selected_decoder == 'QBP-SF2':  
        current_decoder = 'Check-SF2'
    if selected_decoder == 'QBP-SF3':  
        current_decoder = 'Check-SF3'
    restored_info = GL.logistic_setting(current_decoder)
    restored_info[-1] = 'latest'
    check_model, _ = restore_simple_model(current_decoder,restored_info)
    NN_model.decoder_layer.check_model = check_model
    tf.print(f"\nPost-restoration weight:{NN_model.trainable_variables[-1].numpy():.3f}"  ) 
    return NN_model
    
def post_process_input(Model,snr):
    prefix_str = GL.get_map('prefix_str')
    unit_batch_size = GL.get_map('unit_batch_size')
    code = GL.get_map('code_parameters')
    data_dir,iterator = GL.data_setting(code,unit_batch_size,snr)
    #acquiring erroneous cases with necessary modification or perturbation
    print('\n')
    output_list = Decoder_module.postprocess_training(Model,iterator)
    print(f'\n{output_list[4]}th fetches with FER:{output_list[2]:.4f} BER:{output_list[3]:.4f}')
    if GL.get_map('ALL_ZEROS_CODEWORD_TRAINING'):  
        file_name = f'{prefix_str}-retrain-allzero.tfrecord'
    else:
        file_name = f'{prefix_str}-retrain-nonzero.tfrecord'
    retrain_dir_file = data_dir+file_name
    Decoder_module.save_decoded_data(output_list[0],output_list[1],retrain_dir_file)
    print("Collecting targeted cases of decoding is finished!")
    return output_list[2]

def generate_spae_apprx_training_data(SPAE_model,snr):
    unit_batch_size = GL.get_map('unit_batch_size')
    expanding_factor = GL.get_map('num_shifts')*3
    code = GL.get_map('code_parameters')
    data_dir,iterator = GL.data_setting(code,unit_batch_size,snr)
    buffer_inputs_list = []
    buffer_outputs_list = []
    #query of size of input feedings
    input_list = list(iterator.as_numpy_iterator())
    num_counter = len(input_list) 
    num_samples = 0
    for i in range(num_counter):
        if (i+1) % 100 == 0:
            print("Total ",i+1," batches are processed!")
        inputs = input_list[i]
        output_list = SPAE_model(inputs)  
        vc_tensor = output_list[2].stack()
        cv_tensor = output_list[3].stack()
        # buffer_inputs_list.append(vc_tensor)
        # buffer_outputs_list.append(cv_tensor)
        reduced_vc_tensor = vc_tensor[:,:vc_tensor.shape[1]//expanding_factor,:,:]
        reduced_cv_tensor = cv_tensor[:,:cv_tensor.shape[1]//expanding_factor,:,:]
        buffer_inputs_list.append(reduced_vc_tensor)
        buffer_outputs_list.append(reduced_cv_tensor)
        num_samples += inputs[0].shape[0]
    stacked = tf.stack(buffer_inputs_list, axis=0)
    transposed = tf.transpose(stacked, perm=[1, 0, 2, 3, 4])
    buffer_inputs = tf.reshape(transposed, (transposed.shape[0], -1, transposed.shape[-1]))

    stacked = tf.stack(buffer_outputs_list, axis=0)
    transposed = tf.transpose(stacked, perm=[1, 0, 2, 3, 4])
    buffer_outputs = tf.reshape(transposed, (transposed.shape[0], -1, transposed.shape[-1]))
    save_vcv_training_data(buffer_inputs,buffer_outputs,data_dir)
    print("Collecting task is finished!") 
#save training data for check node approximation
def save_vcv_training_data(buffer_inputs,buffer_outputs,data_dir):
    num_iterations = GL.get_map('num_iterations')
    if GL.get_map('ALL_ZEROS_CODEWORD_TRAINING'):  
        distinct_string = 'allzero'
    else:
        distinct_string = 'nonzero'
    for iteration in range(num_iterations):
        output_file_name = f'iteration-{iteration}-{distinct_string}.tfrecord'
        iteration_delimenated_inputs = buffer_inputs[iteration]
        iteration_delimenated_outputs = buffer_outputs[iteration]
        features_matrix,labels_matrix = forming_training_pairs(iteration_delimenated_inputs,iteration_delimenated_outputs) 
        complete_dir_file = data_dir+output_file_name
        print(f"Retraining  data with { iteration_delimenated_inputs.shape[0]} cases to be stored for iteration-{iteration}")
        Data_gen.make_tfrecord((features_matrix, labels_matrix),complete_dir_file)
        print(f" Retraining data of iteration-{iteration} is finished!")

def extract_with_mask(inputs,ground_truths):
    code = GL.get_map('code_parameters')
    mask = code.H
    num_nonzero_per_row = tf.reduce_sum(mask[0])
    num_samples = inputs.shape[0]//mask.shape[0]
    tiled_mask = tf.tile(mask,[num_samples,1])
    extracted_inputs = tf.boolean_mask(inputs, tiled_mask)
    extracted_truths = tf.boolean_mask(ground_truths, tiled_mask)
    compressed_inputs = tf.reshape(extracted_inputs, (-1, num_nonzero_per_row))
    compressed_truths = tf.reshape(extracted_truths, (-1, num_nonzero_per_row))
    return compressed_inputs,compressed_truths

def forming_training_pairs(input_tensor,output_tensor):
    compressed_input,compressed_output = extract_with_mask(input_tensor,output_tensor)
    selected_decoder = GL.get_map('selected_decoder_type')
    if selected_decoder in ['Check-SF1','Check-SF2','Check-SF3']:
        sorted_indices = tf.argsort(-tf.abs(compressed_input), axis=-1)
        features = tf.gather(compressed_input, sorted_indices, batch_dims=1)
        labels = tf.gather(compressed_output, sorted_indices, batch_dims=1)
    return features, labels
