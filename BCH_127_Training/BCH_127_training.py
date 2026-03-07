# -*- coding: utf-8 -*-
import time
T1 = time.time()
import numpy as np
np.set_printoptions(precision=3)
#import matplotlib
import sys
import globalmap as GL
import training_stage as Training_module
 
# Belief propagation using TensorFlow.Run as follows:
    
sys.argv = "python 2.0 2.0 10 1000 20 BCH_127_64_10_strip.alist SPA".split()
#sys.argv = "python 2.8 4.0 25 8000 10 wimax_1056_0.83.alist ANMS".split() 
#setting a batch of global parameters
GL.global_setting(sys.argv)  
selected_decoder_type = GL.get_map('selected_decoder_type')
if GL.get_map('enhanced_NMS_indicator') or selected_decoder_type == 'SPA' : 
    Drawing_EXIT_only = GL.get_map('Drawing_EXIT_only')
    if Drawing_EXIT_only:
        compare_pattern = 'iteration-20_num_shifts-5-sample_size-3000*.pkl'
        save_path = './figures/MI_trajectories'
        Training_module.Draw_multiple_data_file(compare_pattern,save_path)
    else:
        #initial setting for restoring model
        restore_info = GL.logistic_setting()
        #training for the NMS optimization
        #instance of Model creation   
        NMS_model = Training_module.training_stage(restore_info)
        probe_MI = GL.get_map('probe_MI')
        if probe_MI:
            Training_module.post_process_MI(NMS_model)
        else:        
            Training_module.post_process_input(NMS_model)
    #2nd phase to obtain conventional NMS failure data
else: 
    #initial setting for restoring model
    original_NMS_indicator = 'True'
    restore_info = GL.logistic_setting(original_NMS_indicator)
    #training for the NMS optimization
    #instance of Model creation   
    NMS_model = Training_module.training_stage(restore_info,original_NMS_indicator) 
    Training_module.post_process_input(NMS_model,original_NMS_indicator)
T2 =time.time()
print('Running time:%s seconds!'%(T2 - T1))