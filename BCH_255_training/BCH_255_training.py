# -*- coding: utf-8 -*-
import time
T1 = time.time()
import numpy as np
np.set_printoptions(precision=3)
import sys
import globalmap as GL
import training_stage as Training_module
import ms_decoder_dense as Decoder_module
import merger_functions as Merger
from pathlib import Path


sys.argv = "python 3.5 6.0 6 20 1000 10 BCH_255_239_strip.alist QBP-SF2".split()  #Prior to running QBP-S, make sure SPA-1 model has been trained ready.
SNRs = GL.global_setting(sys.argv)  
selected_decoder = GL.get_map('selected_decoder_type')

training_model_phase = GL.get_map('training_model_phase') 
collect_failure_phase = GL.get_map('collect_failure_phase')  
   
if training_model_phase:
    if selected_decoder == 'NMS-1':
        for i,snr in enumerate(SNRs):
            print(f'\nSNR={snr}dB Training of {selected_decoder} starts off:')
            restored_info = GL.logistic_setting(selected_decoder,snr)
            NN_model,train_info = Training_module.restore_saved_model(selected_decoder,restored_info)  
            NN_model = Decoder_module.training_block(NN_model,train_info)                           
    if selected_decoder in ['SPA-1','QBP-SF1','QBP-SF2','QBP-SF3']:
        #initial setting for restoring model
        for i,snr in enumerate(SNRs):
            print(f'\nSNR={snr}dB Training of {selected_decoder} starts off:')
            noise_variance_list = GL.get_map('noise_variance_list')
            GL.set_map('noise_variance', noise_variance_list[i])
            restored_info = GL.logistic_setting(selected_decoder,snr)
            NN_model,train_info = Training_module.restore_saved_model(selected_decoder,restored_info)  
            NN_model = Decoder_module.training_block(NN_model,train_info)           
    if selected_decoder in ['Check-SF1','Check-SF2','Check-SF3']:
        if GL.get_map('generate_check_data'):
            current_decoder = 'SPA-1'
            noise_variance_list = GL.get_map('noise_variance_list')
            for i,snr in enumerate(SNRs):    
                print(f'\nSNR={snr}dB collecting for check NN starts off:')
                GL.set_map('noise_variance', noise_variance_list[i])
                restored_info = GL.logistic_setting(current_decoder,snr)
                SPAE_model,_ = Training_module.restore_saved_model(current_decoder,restored_info)
                Training_module.generate_spae_apprx_training_data(SPAE_model,snr)
        if GL.get_map('global_training'):
            code = GL.get_map('code_parameters')
            num_iterations = GL.get_map('num_iterations')
            unit_batch_size = GL.get_map('unit_batch_size')
            gen_data_dir, _ = GL.data_setting(code,unit_batch_size)
            if GL.get_map('ALL_ZEROS_CODEWORD_TRAINING'):  
                distinct_string = 'allzero'
            else:
                distinct_string = 'nonzero'
            output_file_name =f'merged_train-{distinct_string}.tfrecord'
            decoder_str = 'Check-SF'
            output_full_path = gen_data_dir + f'{decoder_str}/{num_iterations}th/'+output_file_name
            max_iteration_check_data = GL.get_map('max_iteration_check_data')
            result = Merger.check_and_merge(
                root_dir= gen_data_dir,
                keyword='iteration-', 
                output_file = output_full_path,
                overwrite = False,
                max_iteration = max_iteration_check_data
            )
            if result:
                print(f"Merge completed successfully to: {result}")
            else:
                print("Merge was cancelled or failed")
            restored_info = GL.logistic_setting(selected_decoder)
            check_model,train_info = Training_module.restore_saved_model(selected_decoder,restored_info)
            #check_model = Training_module.freeze_model(check_model)
            check_model = Decoder_module.training_block_check_approx(check_model,selected_decoder,train_info)    
        else:
            for i,snr in enumerate(SNRs):  
                print(f'\nSNR={snr}dB Training for check NN starts off:')
                noise_variance_list = GL.get_map('noise_variance_list')              
                GL.set_map('noise_variance', noise_variance_list[i])
                restored_info = GL.logistic_setting(selected_decoder,snr)
                check_model,train_info = Training_module.restore_saved_model(selected_decoder,restored_info)
                #check_model = Training_module.freeze_model(check_model)
                check_model = Decoder_module.training_block_check_approx(check_model,selected_decoder,train_info)    

if collect_failure_phase:
    if GL.get_map('merge_retrain_data'):
        code = GL.get_map('code_parameters')
        prefix_str = GL.get_map('prefix_str')
        num_iterations = GL.get_map('num_iterations')
        unit_batch_size = GL.get_map('unit_batch_size')
        gen_data_dir, _ = GL.data_setting(code,unit_batch_size)
        if GL.get_map('ALL_ZEROS_CODEWORD_TRAINING'):  
            distinct_string = 'allzero'
        else:
            distinct_string = 'nonzero'
        output_file_name =f'{prefix_str}-retrain-{distinct_string}.tfrecord'
        output_full_path = gen_data_dir + f'{selected_decoder}/{num_iterations}th/'+output_file_name
        result = Merger.check_and_merge(
            root_dir= gen_data_dir,
            keyword='retrain-', 
            output_file = output_full_path,
            overwrite = True,
        )
        if result:
            print(f"Merge completed successfully to: {result}")
        else:
            print("Merge was cancelled or failed")
    else:
        FER_list = []
        for i,snr in enumerate(SNRs):
            print(f'\nSNR={snr}dB Training data collecting starts off:')
            if selected_decoder in ['SPA-1','QBP-SF1','QBP-SF2','QBP-SF3']:
                noise_variance_list = GL.get_map('noise_variance_list')
                GL.set_map('noise_variance', noise_variance_list[i])
            restored_info = GL.logistic_setting(selected_decoder,snr)
            NN_model,_ = Training_module.restore_saved_model(selected_decoder,restored_info) 
            FER = Training_module.post_process_input(NN_model,snr)            
            FER_list.append((snr,round(FER.numpy(), 4))) 
        print(f'FER:{FER_list}')
        path = Path('./log/FER_performance.txt')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as f:
            f.write(f'FER:{FER_list}' + '\n')                      
T2 =time.time()
print('Running time:%s seconds!'%(T2 - T1))