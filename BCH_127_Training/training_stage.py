# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 21:29:58 2022

@author: lgw
"""
import numpy as np
np.set_printoptions(precision=3)
#import matplotlib
import tensorflow  as tf
import globalmap as GL
#import ms_decoder_dense as Decoder_module
import ms_decoder_dense as Decoder_module
from typing import Any, Dict,Optional, Union
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
import pickle,os
from sklearn.utils import resample
import nn_net as NN_structure
import scatter_exit as SE
def try_count_flops(model: Union[tf.Module, tf.keras.Model],
                    inputs_kwargs: Optional[Dict[str, Any]] = None,
                    output_path: Optional[str] = None):
    """Counts and returns model FLOPs.
  Args:
    model: A model instance.
    inputs_kwargs: An optional dictionary of argument pairs specifying inputs'
      shape specifications to getting corresponding concrete function.
    output_path: A file path to write the profiling results to.
  Returns:
    The model's FLOPs.
  """
    if hasattr(model, 'inputs'):
        try:
            # Get input shape and set batch size to 1.
            if model.inputs:
                inputs = [
                    tf.TensorSpec([1] + input.shape[1:], input.dtype)
                    for input in model.inputs
                ]
                concrete_func = tf.function(model).get_concrete_function(inputs)
            # If model.inputs is invalid, try to use the input to get concrete
            # function for model.call (subclass model).
            else:
                concrete_func = tf.function(model.call).get_concrete_function(
                    **inputs_kwargs)
            frozen_func, _ = convert_variables_to_constants_v2_as_graph(concrete_func)

            # Calculate FLOPs.
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            if output_path is not None:
                opts['output'] = f'file:outfile={output_path}'
            else:
                opts['output'] = 'none'
            flops = tf.compat.v1.profiler.profile(
                graph=frozen_func.graph, run_meta=run_meta, options=opts)
            return flops.total_float_ops
        except Exception as e:  # pylint: disable=broad-except
            print('Failed to count model FLOPs with error %s, because the build() '
                 'methods in keras layers were not called. This is probably because '
                 'the model was not feed any input, e.g., the max train step already '
                 'reached before this run.', e)
            return None
    return None
def print_flops(model):
    flops = try_count_flops(model)
    print(flops/1e3,"K Flops")
    
def training_stage(restore_info,original_NMS_indicator=False):
    unit_batch_size = GL.get_map('unit_batch_size')
    code = GL.get_map('code_parameters')
    train_iterator = GL.build_training_dataset(code,unit_batch_size)
    exponential_decay = GL.optimizer_setting()
    SE_instance = SE.ScatterEXIT()
    if original_NMS_indicator:
        NMS_model = NN_structure.Convention_NMS_model()
    else:
        NMS_model = Decoder_module.Decoding_model(SE_instance)
    # Explicitly build the model with dummy input
    dummy_input_shape = (None, code.n)  # Replace with actual shape
    NMS_model.build(dummy_input_shape)  # ⚠️ This triggers build() in Decoder_Layer
    tf.print("Pre-restoration weight:", NMS_model.trainable_variables)
    optimizer =  tf.keras.optimizers.legacy.Adam(exponential_decay) 
    # save restoring info
    checkpoint = tf.train.Checkpoint(myAwesomeModel=NMS_model, myAwesomeOptimizer=optimizer)
    logger_info = GL.log_setting(restore_info,checkpoint)
    #unpack related info for restoraging
    start_step = 0
    #unpack related info for restoraging
    [ckpts_dir,ckpt_nm,ckpts_dir_par,restore_step] = restore_info  
    if restore_step:
        start_step,ckpt_f = Decoder_module.retore_saved_model(ckpts_dir,restore_step,ckpt_nm)
        status = checkpoint.restore(ckpt_f)
        print("Post-restoration weight:", NMS_model.trainable_variables)
        #status.assert_existing_objects_matched()  
        status.expect_partial()
    if GL.get_map('loss_process_indicator') and not GL.get_map('probe_MI'):
        NMS_model = Decoder_module.training_block(start_step,NMS_model,optimizer,\
                    exponential_decay,train_iterator,logger_info,restore_info)
    return NMS_model
    
def post_process_input(Model,original_NMS_indicator=False):
    prefix_str = GL.get_map('prefix_str')
    unit_batch_size = GL.get_map('unit_batch_size')
    code = GL.get_map('code_parameters')
    data_dir,iterator = GL.data_setting(code,unit_batch_size)
    #acquiring erroneous cases with necessary modification or perturbation
    print('\n')
    buffer_list = Decoder_module.postprocess_training(Model,iterator)
    print(f'{buffer_list[4]}th fetches with FER:{buffer_list[2]:.4f} BER:{buffer_list[3]:.4f}')
    if GL.get_map('ALL_ZEROS_CODEWORD_TRAINING'):  
        file_name = f'{prefix_str}-retrain-allzero.tfrecord'
    else:
        file_name = f'{prefix_str}-retrain-nonzero.tfrecord'
    if original_NMS_indicator:
        file_name = 'convention-'+file_name
    retrain_dir_file = data_dir+file_name
    Decoder_module.save_decoded_data(buffer_list[0],buffer_list[1],retrain_dir_file)
    print("Collecting targeted cases of decoding is finished!")

def post_process_MI(Model):
    threshold_MI = GL.get_map('threshold_MI')
    num_iterations = GL.get_map('num_iterations')
    unit_batch_size = GL.get_map('unit_batch_size')
    code = GL.get_map('code_parameters')
    _,selected_ds = GL.data_setting(code,unit_batch_size)
    ds_iter = iter(selected_ds)
    num_counter = 0
    num_samples = 0
    while True:
        inputs = next(ds_iter)
        _ = Model(inputs)
        num_counter += 1
        num_samples += inputs[0].shape[0]
        if num_samples >= GL.get_map('validate_dataset_size'):
            break
    IA_list, IE_list = Model.decoder_layer.SE.IA,Model.decoder_layer.SE.IE
    num_arrays_per_position = len(IA_list[0])  
    def accumulate_tensors(info_pack_list):
        final_list = []
        for iteration_index in range(num_iterations):
            merged_arrays = []        
            for array_idx in range(num_arrays_per_position):
                tensors_to_stack = []
                for iter_idx in range(num_counter):
                    tensor = info_pack_list[iteration_index+iter_idx*num_iterations][array_idx]
                    tensors_to_stack.append(tensor)            
                stacked_tensor = tf.stack(tensors_to_stack, axis=0)
                merged_arrays.append(stacked_tensor)       
            final_list.append(tuple(merged_arrays)) 
            #print(iteration_index, stacked_tensor.numpy(), f"{np.mean(stacked_tensor[:20]):.3f}/{np.mean(stacked_tensor[-10:]):.3f}")
        return final_list
    final_ia_list = accumulate_tensors(IA_list)
    final_ie_list = accumulate_tensors(IE_list)      
        
    IA_collapsed_edge_realization = [round(float(tf.reduce_mean(final_ia_list[i][2],axis=0)),4)for i in range(len(final_ia_list))]
    IE_collapased_edge_realization = [round(float(tf.reduce_mean(final_ie_list[i][2],axis=0)),4)for i in range(len(final_ie_list))]
    FER_estimate1 = [Model.decoder_layer.SE.proportion_below_threshold(tf.reshape(final_ia_list[i][1],[-1]),threshold_MI)   for i in range(num_iterations)]
    FER_estimate2 = [Model.decoder_layer.SE.proportion_below_threshold(tf.reshape(final_ie_list[i][1],[-1]),threshold_MI)   for i in range(num_iterations)]
    tf.print(f'IA:{IA_collapsed_edge_realization}')
    tf.print(f'IE:{IE_collapased_edge_realization}')    
    tf.print(f'FER Estimate:{FER_estimate1}/{FER_estimate2}') 
    wrapped_data = (final_ia_list,final_ie_list)  
    save_data(wrapped_data,num_samples)
    print("Collecting mutual information task is finished!")

def save_data(wrapped_data,num_samples):
    """
    Save the collected IA and IE data to a file.
    """      
    directory_path = './data_files/'
    snr_lo = GL.get_map('snr_lo')
    num_iterations = GL.get_map('num_iterations')
    num_shifts = GL.get_map('num_shifts')
    code = GL.get_map('code_parameters')
    num_rows = code.H.shape[0]  
    file_name = f'iteration-{num_iterations}_num_shifts-{num_shifts}-sample_size-{num_samples}-num_rows-{num_rows}.pkl'
    save_dir = os.path.join(directory_path, f'{snr_lo}dB')
    os.makedirs(save_dir, exist_ok=True)        
    full_path = os.path.join(save_dir, file_name)    
    with open(full_path, 'wb') as f:
        pickle.dump(wrapped_data, f)
    print(f"Data saved to: {full_path}")
    
def Draw_data_file():
    Model = SE.ScatterEXIT()
    directory_path = './data_files/'
    full_path = Model.acquire_file_info(directory_path)
    Model.plot_density_strips(full_path,save_path='./figures/MI_trajectory',dpi=800)
def Draw_multiple_data_file(compare_pattern,save_path):  
    snr_lo = GL.get_map('snr_lo')
    Model = SE.ScatterEXIT()
    directory_path = './data_files/'+f'{snr_lo}dB/'
    full_compare_pattern = directory_path+compare_pattern
    Model.plot_density_strips_regular_file(full_compare_pattern,save_path)