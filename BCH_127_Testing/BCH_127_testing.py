# -*- coding: utf-8 -*-
import numpy as np
#np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=5)
#import matplotlib
import sys,os,pickle
import globalmap as GL
import ms_test as Decoder_module
import scatter_exit as SE

sys.argv = "python 2.0 4.5 6 20 1000 18 BCH_127_64_10_strip.alist QBP".split()

GL.global_setting(sys.argv)  

decoder_type = GL.get_map('selected_decoder_type')
code = GL.get_map('code_parameters')
n_dims = code.n
k_dims = code.k
prefix_str = GL.get_map('prefix_str')
authentic_fers = [(2.00,0.38692),
            (2.50,0.2024),
            (3.00,0.07708),
            (3.50,0.02269),
            (4.00,0.00444),
            (4.50,0.00054)]
GL.set_map('true_fer_asymptotes',authentic_fers)

if GL.get_map('reacquire_data'):
    SE_instance = SE.ScatterEXIT()
    #initial setting for restoring model
    restore_info = GL.logistic_setting()
    test_Model = GL.retore_saved_model(restore_info,SE_instance)   
    # the training/testing paramters setting when selected_decoder_type= Combination of VD/VS/SL,HD/HS/SL
    unit_batch_size = GL.get_map('unit_batch_size')

    snr_lo = round(GL.get_map('snr_lo'),2)
    snr_hi = round(GL.get_map('snr_hi'),2)
    snr_num = GL.get_map('snr_num')
    SNRs = np.linspace(snr_lo,snr_hi,snr_num)
    log_filename = GL.log_setting()

    #passing the step of restoring decoding with saved parameters for types of SPA or MS
    FER_list = []
    BER_list = []
    False_P_list = []
    False_N_list = []
    
    # Header for console output
    print("\n" + "="*60)
    print(f"{'Performance Metrics Analysis':^60}")
    print("="*60)
    proportion_list = []
    snr_list = []
    average_iterations_list = []
    Drawing_EXIT_only = GL.get_map('Drawing_EXIT_only')
    if Drawing_EXIT_only:
        num_iterations = GL.get_map('num_iterations')
        num_rows = code.H.shape[0]
        num_shifts = GL.get_map('num_shifts')
        validate_dataset_size = GL.get_map('validate_dataset_size')
        #key_attribute = 'num_shifts-'
        key_attribute = 'num_rows-'       
        #key_attribute = 'Others'       
        if key_attribute == 'num_rows-':
            compare_pattern = f'iteration-{num_iterations}_num_shifts-*-sample_size-{validate_dataset_size}-num_rows-{num_rows}.pkl'
            save_path = './figures/MI_trajectories/'
        if key_attribute == 'num_shifts-':
            compare_pattern = f'iteration-{num_iterations}_num_shifts-{num_shifts}-sample_size-{validate_dataset_size}-num_rows-*.pkl'
            save_path = './figures/MI_trajectories/'
        if key_attribute == 'Others':
            compare_pattern = f'iteration-{num_iterations}_num_shifts-{num_shifts}-sample_size-{validate_dataset_size}-num_rows-{num_rows}.pkl'
            save_path = './figures/Density_trajectories/'
        Decoder_module.Draw_multiple_data_file(compare_pattern,save_path,key_attribute)
    else:
        for snr in SNRs:
            snr = round(snr, 2)
            snr_list.append(snr)
            sigma_square = 1. / (2 * (float(code.k)/float(code.n)) * 10**(snr/10))
            GL.set_map('noise_variance', sigma_square)
            GL.set_map('current_snr', snr)
            if GL.get_map('probe_MI'):
                test_Model.decoder_layer.SE.clear()
                Decoder_module.post_process_MI(test_Model)
                continue
            else:
                metric_list = Decoder_module.post_process4_dia_samples(test_Model)
            [average_FER, average_BER, num_samples,ratio_list,average_iterations] = metric_list
            proportion_list.append(ratio_list)
            average_iterations_list.append(round(average_iterations,2))
        
            # Store data with consistent precision
            FER_list.append(float(f"{average_FER.numpy():.5f}"))
            BER_list.append(float(f"{average_BER.numpy():.5f}"))
            
            # Write detailed per-SNR log
            with open(log_filename, 'a+') as f:
                f.write(f"\n{' SNR: '+str(snr)+'dB ':-^60}\n")
                f.write(f"| {'FER:':<15} {average_FER:.5f}\n")
                f.write(f"| {'BER:':<15} {average_BER:.5f}\n")
                f.write(f"{ratio_list}\n")
                f.write("-"*60)
    if GL.get_map('probe_MI') or GL.get_map('Drawing_EXIT_only'):
        print('All data files are saved !!!')
    else:
        # Formatting functions with visual enhancements
        def format_detailed(title, values, snrs, unit="dB"):
            print(f"\n{title:-^60}")
            for snr, val in zip(snrs, values):
                print(f"({snr:>5}{unit}: {val:.5f})")
        
        def format_summary(title, values,snrs):
            print(f"\n{title:-^60}")
            print("[" + ", ".join([f"({x:.2f},{y:.5f})" for x, y in zip(snrs,values)]) + "]")
        
        # Console output
        format_detailed(" FER by SNR ", FER_list, snr_list)
        format_summary(" FER Summary ", FER_list,snr_list)
        
        format_detailed(" BER by SNR ", BER_list, snr_list)
        format_summary(" BER Summary ", BER_list,snr_list)
        
        # Final log file summary
        with open(log_filename, 'a+') as f:
            f.write(f"\n\n{' Final Summary ':=^60}\n")
            f.write("-"*60 + "\n")
            f.write("FER:    [" + ", ".join([f"({x:.2f},{y:.5f})" for x,y in zip(snr_list,FER_list)]) + "]\n")
            f.write("BER:    [" + ", ".join([f"({x:.2f},{y:.5f})" for x,y in zip(snr_list,BER_list)]) + "]\n")
            f.write("avr_it:    [" + ", ".join([f"({x:.2f},{y:.2f})" for x,y in zip(snr_list,average_iterations_list)]) + "]\n")
            f.write("="*60 + "\n")
        data_path = './data/'
        os.makedirs(data_path, exist_ok=True)
        with open(f'{data_path}drawing_{prefix_str}{n_dims}_{k_dims}', 'wb') as f:
            pickle.dump(snr_list, f)
            pickle.dump(proportion_list,f)
            pickle.dump(average_iterations_list,f)
else:
    data_file = f'./data/drawing_{prefix_str}{n_dims}_{k_dims}'
    if not os.path.exists(data_file):
        print(f"The file '{data_file}' does not exist.")
    with open(data_file, 'rb') as f:
        snr_list = pickle.load(f)
        proportion_list = pickle.load(f)
        average_iterations_list = pickle.load(f)    
if not (GL.get_map('probe_MI') or GL.get_map('Drawing_EXIT_only')):
    fig_save_path = './figs/'
    saved_path = Decoder_module.plot_and_save_multiple_snr_proportions(snr_list,average_iterations_list,proportion_list,save_path=fig_save_path)
    print(f'Figure is saved as {saved_path}!')
