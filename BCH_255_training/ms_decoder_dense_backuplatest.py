# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 10:41:33 2022

@author: Administrator
"""
import tensorflow as tf
import globalmap as GL
import data_generating as Data_gen
import numpy as np
import math
#from distfit import distfit    
np.random.seed(0)
class Decoding_model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layer = Decoder_Layer()
    def build(self, input_shape):
        # Ensure the layer is built with the appropriate input shape
        pass
    def call(self, inputs):
        decoding_info = self.layer(inputs)
        return decoding_info   
    def collect_failed_input_output(self,soft_output_list,labels,indices):
        list_length = self.layer.num_iterations + 1
        buffer_inputs = []
        buffer_labels = []
        for i in indices:
            for j in range(list_length):
                buffer_inputs.append(soft_output_list[j][i])    
                buffer_labels.append(labels[i])
        return buffer_inputs,buffer_labels       

    def get_eval1(self,soft_output_list,labels):
        soft_output = soft_output_list[-1]
        tmp = tf.cast(tf.where(soft_output>0,0,1),tf.int64)
        err_batch = tf.where(tmp == labels,0,1)
        err_sum = tf.reduce_sum(err_batch,-1)
        FER_data = tf.where(err_sum!=0,1,0)     
        FER_count = tf.math.count_nonzero(FER_data)
        #identify the indices of undected decoding errors        
        return FER_count 
    
    def get_eval(self,soft_output_list,labels):
        code = GL.get_map('code_parameters')
        H = code.original_H
        soft_output = soft_output_list[-1]
        tmp = tf.cast(tf.where(soft_output>0,0,1),tf.int64)
        syndrome = tf.matmul(tmp,H,transpose_b=True)%2
        index1 = np.nonzero(tf.reduce_sum(syndrome,-1))[0]
        err_batch = tf.where(tmp == labels,0,1)
        err_sum = tf.reduce_sum(err_batch,-1)
        FER_data = tf.where(err_sum!=0,1,0)
        index = np.nonzero(err_sum)[0]
        diff_set = {}
        if set(index1)!=set(index):
            diff_set = set(index)-set(index1)
            #correct_set = set(index2)-diff_set
            #self.print_undeteced(soft_output_list,iteration,diff_set,correct_set)         
        #number of undetected erroneous bits
        BER_data = 0
        for index in diff_set:
          BER_data += tf.reduce_sum(err_batch[index])
        #number of detected erroneous bits for the initial received sequence
        tmp = tf.cast(tf.where(soft_output_list[0]>0,0,1),tf.int64)
        err_batch = tf.where(tmp == labels,0,1)
        FER_indicator = tf.where(tf.reduce_sum(syndrome,-1,keepdims=True)==0,0,1)
        Detected_num_error = tf.reduce_sum(err_batch*FER_indicator)   
        FER_count = tf.math.count_nonzero(FER_data)
        BER_count = BER_data+Detected_num_error
        #identify the indices of undected decoding errors        
        return FER_count, BER_count,len(diff_set),index1  

    # Function to normalize a vector while preserving signs
    def normalize_with_signs(self,v):
        return np.sign(v) * (np.abs(v) / np.linalg.norm(v))

    def create_samples(self,soft_output_list, labels):
        print('.',end=' ')
        label_bool = tf.cast(labels, tf.bool)
        code = GL.get_map('code_parameters')
        H = code.original_H
        list_length = len(soft_output_list)
        soft_output = soft_output_list[-1]
        tmp_hard = tf.cast(tf.where(soft_output>0,0,1),tf.int64)
        syndrome = tf.matmul(tmp_hard,H,transpose_b=True)%2
        #indices of discarded pair
        index1 = np.nonzero(tf.reduce_sum(syndrome,-1))[0] 
        #cared indices to be classifed as positive or negative pair
        index2 = np.where(tf.reduce_sum(syndrome,-1) == 0)[0]
        pairs, pair_labels = [], []
        data_labels = []
        for i in index2:
            output_hard_decision = tf.cast((soft_output[i] < 0),tf.bool)       
            err_indicator = tf.math.logical_xor(output_hard_decision, label_bool[i])
            FER_sign = tf.reduce_any(err_indicator)
            tmp_matrix = tf.reshape([self.normalize_with_signs(soft_output_list[j][i]) for j in range(list_length)],[list_length,-1])
            pairs.append(tmp_matrix)
            if FER_sign:
                label_vector = tf.reshape([0] * list_length,[list_length,1])
                pair_labels.append(label_vector)               
            else:
                label_vector = tf.reshape([1] * list_length,[list_length,1])
                pair_labels.append(label_vector)
            truth_label_matrix = tf.tile(labels[i:i+1],[list_length,1])
            data_labels.append(truth_label_matrix)
        pairs_matrix = tf.concat(pairs,axis=0)
        labels_matrix = tf.concat(pair_labels,axis=0)
        truth_matrix = tf.concat(data_labels,axis=0)
        return pairs_matrix, labels_matrix,truth_matrix,index1  
    
    def create_samples2(self,soft_output_list, labels):
        print('.',end=' ')
        label_bool = tf.cast(labels, tf.bool)
        code = GL.get_map('code_parameters')
        H = code.original_H
        soft_input = soft_output_list[0]
        soft_output = soft_output_list[-1]
        tmp_hard = tf.cast(tf.where(soft_output>0,0,1),tf.int64)
        syndrome = tf.matmul(tmp_hard,H,transpose_b=True)%2
        #indices of discarded pair
        index1 = np.nonzero(tf.reduce_sum(syndrome,-1))[0] 
        #cared indices to be classifed as positive or negative pair
        index2 = np.where(tf.reduce_sum(syndrome,-1) == 0)[0]
        pairs, pair_labels = [], []
        data_labels = []
        for i in index2:
            output_hard_decision = tf.cast((soft_output[i] < 0),tf.bool)       
            err_indicator = tf.math.logical_xor(output_hard_decision, label_bool[i])
            FER_sign = tf.reduce_any(err_indicator)
            pairs.append([self.normalize_with_signs(soft_input[i]),self.normalize_with_signs(soft_output[i])])
            if FER_sign:
                pair_labels.append(0)               
            else:
                pair_labels.append(1)
            data_labels.append(labels[i])
        return tf.convert_to_tensor(pairs), tf.convert_to_tensor(pair_labels),tf.convert_to_tensor(data_labels),index1  
     
class Decoder_Layer(tf.keras.layers.Layer):
    def __init__(self,initial_value = -0.048):
        super().__init__()
        self.decoder_type = GL.get_map('selected_decoder_type')
        self.num_iterations = GL.get_map('num_iterations')
        self.code = GL.get_map('code_parameters')
        self.H = self.code.H
        self.supplement_matrix = tf.cast(tf.expand_dims(1-self.H,0),tf.float32)
        self.initials = initial_value
   #V:vertical H:Horizontal D:dynamic S:Static  /  VSSL: Vertical Static/Dynamic Shared Layer
    def build(self, input_shape):   
        if GL.get_map('selected_decoder_type') in ['NMS-1']:
            self.shared_check_weight = self.add_weight(name='decoder_check_normalized factor',shape=[1],trainable=True,initializer=tf.keras.initializers.Constant(self.initials ))       
    # Code for model call (handles inputs and returns outputs)
    def call(self,inputs):
        soft_input = inputs[0]
        # Step 1: Compute the L2 norm for each row
        #norms = tf.norm(soft_input, ord=2, axis=1, keepdims=True)       
        # Step 2: Divide each row by its corresponding L2 norm
        #soft_input = soft_input / norms
        labels = inputs[1]    
        soft_output_list,loss,label,_ = self.belief_propagation_op(soft_input,labels)
        return soft_output_list,loss,label
    def get_evenly_shifted_integers(self,start,end,count):
      step = (end - start) // count
      base_integers = [start + i * step for i in range(count)]
      # Calculate the maximum possible shift to keep numbers within the range
      max_shift = end-1-base_integers[-1]
      shift = np.random.randint(0, max_shift)  
      shifted_integers = [x + shift for x in base_integers] 
      return shifted_integers
    def aggregate_cyclic_words(self,soft_input):
        #cycling the input from start to end
        soft_input_split = tf.concat([soft_input[:,0::2],soft_input[:,1::2]],axis=1)
        permutated_codewords = self.frobenius_automorphism(soft_input,soft_input.shape[1])
        soft_input = tf.concat([soft_input,soft_input_split,permutated_codewords],0)
        num_shifts = GL.get_map('num_shifts')
        shifted_integers = self.get_evenly_shifted_integers(0,soft_input.shape[1],num_shifts)
        shifted_input_list = [tf.roll(soft_input,shifted_integers[i],axis=1)  for i in range(num_shifts)]
        super_inputs = tf.concat(shifted_input_list,axis=0)
        return super_inputs,shifted_integers      
      
                
# builds a belief propagation TF graph

    def belief_propagation_op(self,soft_input,labels):
        soft_output_list = [soft_input]*(self.num_iterations+1)
        return tf.while_loop(
            self.continue_condition, # iteration < max iteration?
            self.belief_propagation_iteration, # compute messages for this iteration
            loop_vars = [
                soft_output_list, # soft input for this iteration
                0.,# loss
                labels,
                0, # iteration number
            ]
            )
            
    # compute messages from variable nodes to check nodes
    def compute_vc(self,soft_input):
        check_matrix_H = tf.cast(self.H,tf.float32)                                      
        vc_matrix = tf.expand_dims(soft_input,axis=1)*check_matrix_H
        return vc_matrix  
 
    # compute messages from check nodes to variable nodes
    def compute_cv(self,vc_matrix):
        check_matrix_H = self.H
        #operands sign processing 
        sign_info = self.supplement_matrix + vc_matrix
        vc_matrix_sign = tf.sign(sign_info)
        temp = tf.reduce_prod(vc_matrix_sign,2,keepdims=True)
        transition_sign_matrix = temp*check_matrix_H
        result_sign_matrix = transition_sign_matrix*vc_matrix_sign 
        #preprocessing data for later calling of top k=2 largest items
        back_matrix = tf.where(check_matrix_H==0,-1e20-1,0.)
        back_matrix = tf.expand_dims(back_matrix,0)
        vc_matrix_abs = tf.abs(vc_matrix)
        vc_matrix_abs_clip = tf.clip_by_value(vc_matrix_abs, 0, 1e20)
        vc_matrix_abs_minus = -tf.abs(vc_matrix_abs_clip)
        vc_decision_matrix = vc_matrix_abs_minus+back_matrix
        min_submin_info = tf.nn.top_k(vc_decision_matrix,k=2)
        min_column_matrix = -min_submin_info[0][:,:,0:1]
        min_column_matrix = min_column_matrix*check_matrix_H
        second_column_matrix = -min_submin_info[0][:,:,1:2]
        second_column_matrix = second_column_matrix*check_matrix_H  
        result_matrix = tf.where(vc_matrix_abs_clip>min_column_matrix,min_column_matrix,second_column_matrix)          
        if GL.get_map('selected_decoder_type') in ['NMS-1']:
          normalized_tensor = tf.nn.softplus(self.shared_check_weight) 
        cv_matrix = normalized_tensor*result_matrix*tf.stop_gradient(result_sign_matrix)         
        return cv_matrix

    
    def calculation_loss(self,soft_output,labels):
         #cross entroy
        labels = tf.cast(labels,tf.float32)
        CE_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=-soft_output, labels=labels)) 
        return CE_loss   
    #combine messages to get posterior LLRs
    def marginalize(self,cv_matrix, soft_input,shifted_integers,labels,iteration,soft_output_list):
        #tmp_labels = self.interleave_columns(labels[:,::2],labels[:,1::2]) 
        # print(np.sum(self.code.H@tf.transpose(tmp_labels)%2))
        # print(np.sum(self.code.H@tf.transpose(labels)%2))
        temp = tf.reduce_sum(cv_matrix,1)
        #aligning with cycled input
        num_shifts = GL.get_map('num_shifts')
        batch_size = soft_input.shape[0]//num_shifts
        basic_batch_size = batch_size//3 #three kinds of permutations
        # Use vectorized operations
        shifted_temps = [tf.roll(temp[i*batch_size:(i+1)*batch_size], shift=-shifted_integers[i], axis=1) for i in range(num_shifts)]
        #shift permutation
        shift_list = [shifted_temps[i][:basic_batch_size] for i in range(num_shifts)]
        tensor1 = tf.add_n(shift_list)
        shift2_list = [shifted_temps[i][basic_batch_size:2*basic_batch_size] for i in range(num_shifts)]
        tensor2 = tf.add_n(shift2_list)
        width = math.ceil(tensor2.shape[1]/2)
        interleaved = self.interleave_columns(tensor2[:,:width],tensor2[:,width:]) 
        shift3_list = [shifted_temps[i][2*basic_batch_size:] for i in range(num_shifts)]
        tensor3 = tf.add_n(shift3_list)
        permutated = self.inverse_frobenius_automorphism(tensor3, tensor3.shape[1])
        soft_output = soft_output_list[iteration]+(tensor1+interleaved+permutated)/(3*num_shifts)
        # Step 1: Compute the L2 norm for each row
        #norms = tf.norm(soft_output, ord=2, axis=1, keepdims=True)       
        # Step 2: Divide each row by its corresponding L2 norm
        #soft_output = soft_output / norms
        soft_output_list[iteration+1] = soft_output
        return soft_output 
    
        
    def interleave_columns(self,tensor1, tensor2):
        """
        Interleave the columns of two tensors into one tensor.
        
        Args:
            tensor1 (tf.Tensor): First tensor of shape (m, n).
            tensor2 (tf.Tensor): Second tensor of shape (m, n) or (m, n-1).
        
        Returns:
            tf.Tensor: Tensor with interleaved columns.
        """
        # Ensure tensors have compatible shapes
        if tensor1.shape[0] != tensor2.shape[0]:
            raise ValueError("Tensors must have the same number of rows")
        
        # Get shapes
        m, n1 = tensor1.shape
        n2 = tensor2.shape[1]
        
        # Interleave columns using stack and reshape
        if n1 == n2:
            # If the number of columns is the same, stack along the third axis and then reshape
            interleaved = tf.reshape(tf.stack([tensor1, tensor2], axis=2), (m, n1 + n2))
        elif n1 == n2 + 1:
            # If tensor1 has one more column than tensor2
            tensor1_split = tf.split(tensor1, [n2, 1], axis=1)
            interleaved = tf.reshape(tf.stack([tensor1_split[0], tensor2], axis=2), (m, n1 + n2 - 1))
            interleaved = tf.concat([interleaved, tensor1_split[1]], axis=1)
        else:
            raise ValueError("tensor1 must have the same number of columns or one more column than tensor2")
        
        return interleaved
    
    def frobenius_automorphism(self,codeword, n):
        """
        Apply the Frobenius automorphism to a binary codeword using the formula (2 * i) % n.
        
        Args:
            codeword (tf.Tensor): Binary codeword of shape (1, n).
            n (int): Length of the codeword.
        
        Returns:
            tf.Tensor: Permuted codeword.
        """
        # Generate positions using the formula (2 * i) % n
        positions = [(2 * i) % n for i in range(n)]
        
        # Create the permutation matrix using the generated positions
        perm_matrix = tf.eye(n, dtype=codeword.dtype)
        perm_matrix = tf.gather(perm_matrix, positions, axis=1)
        
        # Apply the permutation
        permuted_codeword = tf.matmul(codeword, perm_matrix)
        
        return permuted_codeword
    
    def inverse_frobenius_automorphism(self,codeword, n):
        """
        Apply the inverse Frobenius automorphism to a binary codeword using the inverse of (2 * i) % n.
        
        Args:
            codeword (tf.Tensor): Binary codeword of shape (1, n).
            n (int): Length of the codeword.
        
        Returns:
            tf.Tensor: Permuted codeword (original codeword before applying the Frobenius automorphism).
        """
        # Generate positions using the formula (2 * i) % n
        positions = [(2 * i) % n for i in range(n)]
        
        # Generate the inverse positions by reversing the permutation
        inverse_positions = [positions.index(i) for i in range(n)]
        
        # Create the permutation matrix for the inverse positions
        perm_matrix = tf.eye(n, dtype=codeword.dtype)
        perm_matrix = tf.gather(perm_matrix, inverse_positions, axis=1)
        
        # Apply the inverse permutation
        recovered_codeword = tf.matmul(codeword, perm_matrix)
        
        return recovered_codeword
    
    def continue_condition(self,soft_output_list,loss,labels,iteration):
        condition = (iteration < self.num_iterations) 
        return condition
    
    def belief_propagation_iteration(self,soft_output_list,loss,labels, iteration):
        # compute vc
        super_input,shifted_integers = self.aggregate_cyclic_words(soft_output_list[iteration])
        vc_matrix = self.compute_vc(super_input)
        # compute cv
        cv_matrix = self.compute_cv(vc_matrix)      
        # get output for this iteration
        soft_output = self.marginalize(cv_matrix, super_input,shifted_integers,labels,iteration,soft_output_list)  
        iteration += 1   
        loss = self.calculation_loss(soft_output,labels)
        return soft_output_list, loss,labels, iteration

def retore_saved_model(restore_ckpts_dir,restore_step,ckpt_nm):
    print("Ready to restore a saved latest or designated model!")
    ckpt = tf.train.get_checkpoint_state(restore_ckpts_dir)
    if ckpt and ckpt.model_checkpoint_path: # ckpt.model_checkpoint_path means the latest ckpt
      if restore_step == 'latest':
        ckpt_f = tf.train.latest_checkpoint(restore_ckpts_dir)
        start_step = int(ckpt_f.split('-')[-1]) 
      else:
        ckpt_f = restore_ckpts_dir+ckpt_nm+'-'+restore_step
        start_step = int(restore_step)
      print('Loading wgt file: '+ ckpt_f)   
    else:
      print('Error, no qualified file found')
    return start_step,ckpt_f
#save modified data for postprocessing
def save_decoded_data(buffer_inputs,buffer_labels,dir_file):
    #code = GL.get_map('code_parameters')
    stacked_buffer_info = tf.stack(buffer_inputs)
    stacked_buffer_label = tf.stack(buffer_labels)
    print("\nData for retraining  with %d cases to be stored " % stacked_buffer_info.shape[0])
    data = (stacked_buffer_info.numpy(),stacked_buffer_label.numpy())
    Data_gen.make_tfrecord(data, out_filename=dir_file)    
    print("Data storing finished!")

def calculate_loss(inputs,labels):
    labels = tf.cast(labels,tf.float32)  
    #measure discprepancy via cross entropy metric which acts as the loss definition for deep learning per batch         
    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=-inputs, labels=labels))
    return  loss


def calculate_list_cross_entropy_ber(input_list,labels):
    cross_entropy_list = []
    ber_list = []
    for i in range(len(input_list)):
        cross_entropy_element = calculate_loss(input_list[i],labels).numpy()
        cross_entropy_list.append(cross_entropy_element)
        current_hard_decision = tf.where(input_list[i]>0,0,1)
        compare_result = tf.where(current_hard_decision!=labels,1,0)
        num_errors = tf.reduce_sum(compare_result)
        ber_list.append(num_errors)
    return cross_entropy_list,ber_list

# Function to normalize a vector while preserving signs
def normalize_with_signs(v):
    return np.sign(v) * (np.abs(v) / np.linalg.norm(v))

#postprocessing after first stage training
def generate_parity_pass_samples(Model,iterator):
    #collecting erroneous decoding info
    buffer_inputs = []
    buffer_labels = []
    data_labels = []
    #query of size of input feedings
    input_list = list(iterator.as_numpy_iterator())
    num_counter = len(input_list) 
    for i in range(num_counter):
        if not (i+1) % 100:
            print("Total ",i+1," batches are processed!")
        inputs = input_list[i]
        soft_output_list,_,label = Model(inputs)
        pair,pair_label,data_label,_ = Model.create_samples(soft_output_list,label)
        buffer_inputs.append( pair)
        buffer_labels.append(pair_label)
        data_labels.append(data_label)
    sample_matrix = tf.concat(buffer_inputs,axis=0)
    data_labels_matrix = tf.concat(data_labels,axis=0)
    labels_vector =tf.cast(tf.concat(buffer_labels,axis=0),tf.float32)
    feature_matrix = tf.concat([sample_matrix,labels_vector],1)
    return feature_matrix,data_labels_matrix
#postprocessing after first stage training
def generate_distance_model_samples(Model,iterator):
    #collecting erroneous decoding info
    buffer_inputs = []
    buffer_labels = []
    data_labels = []
    #query of size of input feedings
    input_list = list(iterator.as_numpy_iterator())
    num_counter = len(input_list) 
    for i in range(num_counter):
        if not (i+1) % 100:
            print("Total ",i+1," batches are processed!")
        inputs = input_list[i]
        soft_output_list,_,label = Model(inputs)
        pair,pair_label,data_label,_ = Model.create_samples(soft_output_list,label)
        buffer_inputs.append( pair)
        buffer_labels.append(pair_label)
        data_labels.append(data_label)
    sample_matrix = tf.concat(buffer_inputs,axis=0)
    data_labels_matrix = tf.concat(data_labels,axis=0)
    labels_vector = tf.concat(buffer_labels,axis=0) 
    return sample_matrix,labels_vector,data_labels_matrix
   
def postprocess_statistics(Model,iterator):
    #query of size of input feedings
    input_list = list(iterator.as_numpy_iterator())
    num_counter = len(input_list) 
    FER_sum = 0.
    size_sum = 0
    for i in range(num_counter):
        print('.',end='')
        inputs = input_list[i]
        soft_output_list,_,label = Model(inputs)
        current_size = label.shape[0]
        #print(Model.trainable_variables)
        FER_count = Model.get_eval1(soft_output_list,label)
        FER_sum += FER_count
        size_sum += current_size
        #tf.print(FER,Model.trainable_variables)
        if (i+1) % 10:
            print("Total ",i+1," batches are processed!")
            print(f'FER:{FER_sum/size_sum:.4f}')
    return FER_sum/size_sum,num_counter

#postprocessing after first stage training
def postprocess_training(Model,iterator):
    #collecting erroneous decoding info
    buffer_inputs = []
    buffer_labels = []
    #query of size of input feedings
    input_list = list(iterator.as_numpy_iterator())
    num_counter = len(input_list) 
    FER_sum = 0
    BER_sum = 0
    size_sum = 0
    for i in range(num_counter):
        if not (i+1) % 100:
            print("Total ",i+1," batches are processed!")
        inputs = input_list[i]
        soft_output_list,_,label = Model(inputs)
        current_size = label.shape[0]
        FER_count,BER_count,_,indices= Model.get_eval(soft_output_list,label)
        buffer_inputs_tmp,buffer_labels_tmp = Model.collect_failed_input_output(soft_output_list,label,indices)   
        buffer_inputs.append(buffer_inputs_tmp)
        buffer_labels.append(buffer_labels_tmp)
        FER_sum += FER_count
        BER_sum += BER_count
        size_sum += current_size
    buffer_inputs = [j for i in buffer_inputs for j in i]
    buffer_labels = [j for i in buffer_labels for j in i]
    n_dims = label.shape[1]
    return buffer_inputs,buffer_labels,FER_sum/size_sum,BER_sum/(size_sum*n_dims),num_counter
    
#main training process
def training_block(Model,optimizer,exponential_decay,selected_ds,log_info,restore_info,batch_index):
    #query of size of input feedings
    input_list = list(selected_ds.as_numpy_iterator())
    num_counter = len(input_list) 
    summary_writer,manager_current = log_info
    ckpts_dir,ckpt_nm,ckpts_dir_par,_= restore_info
    if batch_index < GL.get_map('termination_step'):
        for i in range(num_counter):
                with tf.GradientTape() as tape:
                    inputs = input_list[i]
                    soft_output_list,loss,label = Model(inputs)
                    fer_count,ber_count,_,_= Model.get_eval(soft_output_list,label)
                    fer = fer_count/label.shape[0]
                    ber = ber_count/(label.shape[0]*label.shape[1])
                    #fer = Model.get_eval(soft_output_list,label)
                    grads = tape.gradient(loss,Model.variables)
                grads_and_vars=zip(grads, Model.variables)
                capped_gradients = [(tf.clip_by_norm(grad,1e2), var) for grad, var in grads_and_vars if grad is not None]
                #capped_gradients = [(tf.clip_by_value(grad,-1,1), var) for grad, var in grads_and_vars if grad is not None]
                optimizer.apply_gradients(capped_gradients)
                with summary_writer.as_default():                               # the logger to be used
                  tf.summary.scalar("loss", loss, step=batch_index)
                  tf.summary.scalar("FER", fer, step=batch_index)  # you can also add other indicators below
                  tf.summary.scalar("BER", ber, step=batch_index)  # you can also add other indicators below     
                # log to stdout 
                print_interval = GL.get_map('print_interval')
                record_interval = GL.get_map('record_interval')
                if batch_index % print_interval== 0: 
                    tf.print("Step%4d: Lr:%.3f Ls:%.1f FER:%.3f BER:%.4f"%\
                    (batch_index,exponential_decay(batch_index),loss, fer, ber ))  
                if batch_index % record_interval == 0:
                    print("For all layers at the %4d-th step:"%batch_index)
                    manager_current.save(checkpoint_number=batch_index)
                    for variable in Model.variables:
                        print(str(variable.numpy()))  
                    with open(ckpts_dir_par+'values.txt','a+') as f:
                        f.write("For all layers at the %4d-th step:\n"%batch_index)
                        for variable in Model.variables:
                            f.write(variable.name+' '+str(variable.numpy())) 
                        f.write('\n')  
                batch_index += 1
                if batch_index > GL.get_map('termination_step'):
                    break
    else:
        inputs = input_list[0]
        _ = Model(inputs)
    print("Final selected parameters:")
    for weight in  Model.layer.get_weights():
      print(weight)
    return Model           