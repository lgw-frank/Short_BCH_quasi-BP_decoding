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
from tensorflow import keras 
from tensorflow.keras import  layers
     
class conv_sf1_bitwise(keras.Model):
    def __init__(self,input_width):
        super(conv_sf1_bitwise, self).__init__()   
        self.input_width = input_width 
        self.conv1 = layers.Conv1D(16, 3, padding='valid', activation=HardSwish())
        self.conv2 = layers.Conv1D(8, 3, padding='valid', activation=HardSwish())
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(8, activation=HardSwish())        
        self.output_layer = layers.Dense(self.input_width)
        self.scale = self.add_weight(
            name='scale',
            shape=(),
            initializer=tf.constant_initializer(-2.0),
            trainable=True
        )
        dummy_input = tf.random.uniform((1, self.input_width, 1), dtype=tf.float32)
        self(dummy_input)
            
    def call(self, inputs,training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense(x) 
        x = self.output_layer(x)     
        x = x * tf.nn.softplus(self.scale) 
        if not training:
            x = tf.maximum(x, 0.0)
            x = tf.sort(x, axis=-1, direction='ASCENDING')
        return x
    def loss_query(self, soft_outputs, labels, inputs):
        distance_loss = tf.reduce_mean(tf.reduce_sum(tf.square(soft_outputs - labels),axis=1))
        order_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.relu(soft_outputs[:, :-1]-soft_outputs[:, 1:]),axis=1))   
        positive_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.relu(-soft_outputs),axis=1))
        loss_sum = distance_loss + order_loss + positive_loss
        return loss_sum
    
class conv_sf2_bitwise(keras.Model):
    def __init__(self,input_width):
        super(conv_sf2_bitwise, self).__init__()   
        self.input_width = input_width 
        self.conv1 = layers.Conv1D(4, 3, padding='valid',activation=HardSwish())
        self.conv2 = layers.Conv1D(2, 3, padding='valid',activation=HardSwish())
        self.flatten = layers.Flatten()       
        self.output_layer = layers.Dense(1,activation='linear')
        self.scale = self.add_weight(
            name='scale',
            shape=(),
            initializer=tf.constant_initializer(-2.0),
            trainable=True
        )
        dummy_input = tf.random.uniform((1,self.input_width, 1), dtype=tf.float32)
        self(dummy_input)
            
    def call(self, inputs,training=False):    
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.output_layer(x)    
        x = x * tf.nn.softplus(self.scale) 
        if not training:
          x = tf.maximum(x, 0.0)
          x = tf.sort(x, axis=-1, direction='ASCENDING')
        return x 
    def loss_query(self, soft_outputs, labels,inputs):
        distance_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(soft_outputs - labels),axis=1))       
        positive_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.relu(-soft_outputs),axis=1))
        loss_sum = distance_loss + positive_loss
        return loss_sum
    
class conv_sf3_bitwise(keras.Model):
    def __init__(self,input_width):
        super(conv_sf3_bitwise, self).__init__()   
        self.input_width = input_width
        self.conv1 = layers.Conv1D(4, 3, padding='valid',activation=HardSwish())
        self.conv2 = layers.Conv1D(2, 3, padding='valid',activation=HardSwish())
        self.flatten = layers.Flatten()     
        self.dense = layers.Dense(2, activation=HardSwish())   
        self.output_layer = layers.Dense(1,activation='linear')
        self.scale = self.add_weight(
            name='scale',
            shape=(),
            initializer=tf.constant_initializer(-2.0),
            trainable=True
        )
        dummy_input = tf.random.uniform((self.input_width+1,self.input_width, 1), dtype=tf.float32)
        self(dummy_input)
            
    def call(self, inputs,training=False):    
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.output_layer(x)    
        x = x * tf.nn.softplus(self.scale) 
        x = tf.reshape(x,[-1,self.input_width+1])
        if not training:
          x = tf.maximum(x, 0.0)
          x = tf.sort(x, axis=-1, direction='ASCENDING')
        return x 
    def loss_query(self, soft_outputs, labels,inputs):
        distance_loss = tf.reduce_mean(tf.reduce_sum(tf.square(soft_outputs - labels),axis=1))       
        order_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.relu(soft_outputs[:, :-1]-soft_outputs[:, 1:]),axis=1))   
        positive_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.relu(-soft_outputs),axis=1))
        loss_sum = distance_loss + order_loss + positive_loss
        return loss_sum
  
class HardSwish(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs) 
    def build(self, input_shape):
        super().build(input_shape)  
    def call(self, x): 
        return x * tf.nn.relu6(x + 3.) / 6
     
class BPSK_EM_Estimator:
    def __init__(self, max_iter=100, tol=1e-6):
        self.max_iter = max_iter
        self.tol = tol        
    def gaussian_pdf(self, x, mu, sigma2):
        return tf.exp(-(x - mu)**2 / (2 * sigma2)) / tf.sqrt(2 * np.pi * sigma2)
    
    def fit(self, soft_inputs, verbose=True):
        y = tf.reshape(soft_inputs, [-1])
        n = tf.cast(tf.shape(y)[0], tf.float32)
        y_sorted = tf.sort(y)
        split_idx = tf.cast(n/2, tf.int32)        
        mu_neg_init = tf.reduce_mean(y_sorted[:split_idx])   # 1 -> -1
        mu_pos_init = tf.reduce_mean(y_sorted[split_idx:])   # 0 -> +1   
        sigma_neg2_init = tf.reduce_mean((y_sorted[:split_idx] - mu_neg_init)**2)
        sigma_pos2_init = tf.reduce_mean((y_sorted[split_idx:] - mu_pos_init)**2)
        sigma2_init = (sigma_neg2_init + sigma_pos2_init) / 2
        if mu_neg_init > mu_pos_init:
            mu_neg_init, mu_pos_init = mu_pos_init, mu_neg_init
        mu_neg = tf.Variable(mu_neg_init, dtype=tf.float32)
        mu_pos = tf.Variable(mu_pos_init, dtype=tf.float32)
        sigma2 = tf.Variable(sigma2_init, dtype=tf.float32)   
        pi = 0.5  # prior probability 
        prev_log_likelihood = -np.inf
        
        for iteration in range(self.max_iter):
            # E-Step: posterior probability calculation
            # prob_bit0: 
            prob_bit0 = pi * self.gaussian_pdf(y, mu_pos, sigma2)
            # prob_bit1: 
            prob_bit1 = (1-pi) * self.gaussian_pdf(y, mu_neg, sigma2)
            gamma_bit0 = prob_bit0 / (prob_bit0 + prob_bit1 + 1e-10)
            gamma_bit1 = 1 - gamma_bit0
            log_likelihood = tf.reduce_sum(tf.math.log(prob_bit0 + prob_bit1 + 1e-10))
            # M-Step: udpate parameters
            n_bit0 = tf.reduce_sum(gamma_bit0)
            n_bit1 = tf.reduce_sum(gamma_bit1)         
            # update means
            new_mu_pos = tf.reduce_sum(gamma_bit0 * y) / (n_bit0 + 1e-10)   # 比特0的均值
            new_mu_neg = tf.reduce_sum(gamma_bit1 * y) / (n_bit1 + 1e-10)   # 比特1的均值
            
            # udpate the sole variance
            new_sigma2 = (tf.reduce_sum(gamma_bit0 * (y - new_mu_pos)**2) + 
                         tf.reduce_sum(gamma_bit1 * (y - new_mu_neg)**2)) / n            
            # apply updates
            mu_pos.assign(new_mu_pos)
            mu_neg.assign(new_mu_neg)
            sigma2.assign(new_sigma2)           
            # check convergence
            if iteration > 0:
                if tf.abs(log_likelihood - prev_log_likelihood) < self.tol:
                    if verbose:
                        print(f"converging at {iteration}-th iteration")
                    break           
            prev_log_likelihood = log_likelihood           
            if verbose and iteration % 10 == 0:
                print(f"Iter {iteration}: mu_neg={mu_neg.numpy():.4f} (bit 1), "
                      f"mu_pos={mu_pos.numpy():.4f} (bit 0), "
                      f"sigma2={sigma2.numpy():.4f}, logL={log_likelihood.numpy():.2f}")       
        return {
            'mu_bit0': mu_pos.numpy(),      # 0 -> +1
            'mu_bit1': mu_neg.numpy(),      # 1 -> -1
            'sigma2': sigma2.numpy()
        }

class SPAE_model(tf.keras.Model):
    def __init__(self,check_model=None,se_instance=None,initial=None):
        super().__init__()
        self.decoder_layer = SPAE_layer(check_model,se_instance,initial)  # Explicitly track the layer
    def build(self, input_shape):
        if hasattr(input_shape, 'as_list'):
            processed_shape = input_shape.as_list()
        else:
            processed_shape = list(input_shape)       
        # Build your decoder layer with concrete dimensions
        self.decoder_layer.build(processed_shape)       
        # Skip super().build() entirely - it's often not needed
        self.built = True  # Manually mark as built
    def call(self,inputs): 
        output_list = self.decoder_layer(inputs)
        return output_list 
    def get_eval(self,soft_output_array,labels):
        code = GL.get_map('code_parameters')
        num_iterations = GL.get_map('num_iterations')
        labels = tf.cast(labels,tf.int64)
        H = code.original_H
        soft_output = soft_output_array.read(num_iterations)
        tmp = tf.cast(tf.where(soft_output>0,0,1),tf.int64)
        syndrome = tf.matmul(tmp,H,transpose_b=True)%2
        index1 = np.nonzero(tf.reduce_sum(syndrome,-1))[0]
        err_batch = tf.where(tmp == labels,0,1)
        err_sum = tf.reduce_sum(err_batch,-1)
        BER_count = tf.reduce_sum(err_sum)
        FER_data = tf.where(err_sum!=0,1,0)
        FER_count = tf.math.count_nonzero(FER_data)
        #identify the indices of undected decoding errors        
        return FER_count, BER_count,index1   
  
    def collect_failed_input_output(self,soft_output_array,labels,indices):
        list_length = GL.get_map('num_iterations')+1
        soft_output_list = soft_output_array.stack()
        buffer_inputs = []
        buffer_labels = []
        #indices = tf.squeeze(index,1).numpy()
        for i in indices:
            for j in range(list_length):
                buffer_inputs.append(soft_output_list[j][i])    
                buffer_labels.append(labels[i])
        return buffer_inputs,buffer_labels  
    
class SPAE_layer(tf.keras.layers.Layer):
    def __init__(self,check_model=None,initial=None):
        super().__init__()
        self.decoder_type = GL.get_map('selected_decoder_type')
        self.num_iterations = GL.get_map('num_iterations')
        self.code = GL.get_map('code_parameters')
        self.H = self.code.H
        self.initial = initial if initial is not None else -2.
        self.supplement_matrix =  tf.expand_dims(tf.cast(1-self.H,dtype=tf.float32),0)
        self.check_model = check_model
        self.em = BPSK_EM_Estimator()
        self.decoder_balance_normalizor = self.add_weight(
            name='decoder_balance_normalizor',
            shape=[],
            trainable=True,
            initializer=tf.keras.initializers.Constant(self.initial),
            dtype=tf.float32)      #\eta:aggregation coefficient
    def build(self, input_shape):   
        if self.se:
            self.se.clear()
    def call(self, inputs):
        noise_variance = GL.get_map('noise_variance')
        soft_input = 2/noise_variance*inputs[0]  #transformed int LLR          
        labels = inputs[1]
        outputs = self.belief_propagation_op(soft_input,labels)
        return outputs  # ✅ stack before returning

    def aggregate_cyclic_words(self,soft_input,labels):
        #cycling the input from start to end
        num_shifts = GL.get_map('num_shifts')
        if num_shifts > 0:
            soft_input_split = tf.concat([soft_input[:,0::2],soft_input[:,1::2]],axis=1)
            label_split = tf.concat([labels[:,0::2],labels[:,1::2]],axis=1)
            permutated_sequences = self.frobenius_automorphism(soft_input,soft_input.shape[1])
            permutated_labels = self.frobenius_automorphism(labels,labels.shape[1])            
            soft_input = tf.concat([soft_input,soft_input_split,permutated_sequences],0)  
            new_labels = tf.concat([labels,label_split,permutated_labels],0)  
            shifted_integers = self.get_evenly_shifted_integers(soft_input.shape[1],num_shifts)
            shifted_input_list = [tf.roll(soft_input,shifted_integers[i],axis=1)  for i in range(num_shifts)]
            shifted_label_list = [tf.roll(new_labels,shifted_integers[i],axis=1)  for i in range(num_shifts)]
            super_inputs = tf.concat(shifted_input_list,axis=0)
            super_labels = tf.concat(shifted_label_list,axis=0)
        else:
            super_inputs = soft_input
            super_labels = labels
            shifted_integers = [-1]
        return super_inputs,super_labels,shifted_integers  
                     
    def normalize_input(self, soft_input):
        mean = tf.reduce_mean(tf.abs(soft_input),axis=1,keepdims=True)
        base_line = 2/GL.get_map('noise_variance')
        scale_factor = base_line/mean
        soft_input = soft_input*scale_factor
        return soft_input           
      
    def belief_propagation_op(self, soft_input,labels):
        soft_output_array = tf.TensorArray(
            dtype=tf.float32,
            size=self.num_iterations+1,
            clear_after_read=False  # <-- Required if you want to read an index multiple times
        )
        soft_vc_matrix_array = tf.TensorArray(
            dtype=tf.float32,
            size=self.num_iterations,
            clear_after_read=False  # <-- Required if you want to read an index multiple times
        )
        soft_cv_matrix_array = tf.TensorArray(
            dtype=tf.float32,
            size=self.num_iterations,
            clear_after_read=False  # <-- Required if you want to read an index multiple times
        )
        # Write initial value
        iteration = 0
        loss = 0.
        soft_output_array = soft_output_array.write(iteration, soft_input)
        def condition(soft_output_array,labels,soft_vc_matrix_array,soft_cv_matrix_array,iteration,loss):
            return iteration < self.num_iterations
        def body(soft_output_array,labels,soft_vc_matrix_array,soft_cv_matrix_array,iteration,loss):
            soft_inputs = soft_output_array.read(iteration)
            super_inputs,super_labels,shifted_integers = self.aggregate_cyclic_words(soft_inputs,labels)
            soft_vc_matrix_array = self.compute_vc(super_inputs,soft_vc_matrix_array,iteration)
            # compute cv
            soft_cv_matrix_array = self.compute_cv_spa(soft_vc_matrix_array,soft_cv_matrix_array,iteration) 
            # get output for this iteration
            soft_output = self.marginalize(super_inputs,soft_cv_matrix_array,soft_output_array,shifted_integers,iteration)  
            iteration += 1 
            # rectify the mean of inputs
            soft_output = self.normalize_input(soft_output)
            soft_output_array = soft_output_array.write(iteration, soft_output)
            #loss += self.calculation_loss(soft_output,labels,loss)    
            return labels,soft_output_array,soft_vc_matrix_array,soft_cv_matrix_array,iteration,loss
    
        soft_output_array,labels,soft_vc_matrix_array,soft_cv_matrix_array,iteration,loss = tf.while_loop(
            condition,
            body,
            loop_vars=[soft_output_array,labels,soft_vc_matrix_array,soft_cv_matrix_array,iteration,loss ]
        )   
        return soft_output_array,labels,soft_vc_matrix_array,soft_cv_matrix_array,iteration,loss
    
    def calculation_loss(self,soft_output,labels,loss):
         #cross entroy
        labels = tf.cast(labels,tf.float32)
        CE_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=-soft_output, labels=labels)) 
        return CE_loss+loss
    
    def compute_vc(self,super_input,soft_vc_matrix_array,iteration):
        check_matrix_H = tf.cast(self.code.H,tf.float32)   
        vc_matrix = tf.expand_dims(super_input,axis=1)*check_matrix_H       
        updated_soft_vc_matrix_array = soft_vc_matrix_array.write(iteration, vc_matrix)    
        return updated_soft_vc_matrix_array               
    def compute_cv_reduced(self,soft_vc_matrix_array,soft_cv_matrix_array,iteration):
        vc_matrix = soft_vc_matrix_array.read(iteration)
        check_matrix_H = self.code.H
        #operands sign processing 
        supplement_matrix = tf.cast(1-check_matrix_H,dtype=tf.float32)
        supplement_matrix = tf.expand_dims(supplement_matrix,0)
        sign_info = supplement_matrix + vc_matrix
        vc_matrix_sign = tf.sign(sign_info)
        temp = tf.reduce_prod(vc_matrix_sign,axis=-1)
        temp = tf.expand_dims(temp,axis=-1)
        transition_sign_matrix = temp*check_matrix_H
        result_sign_matrix = transition_sign_matrix*vc_matrix_sign 
        #oprations on magnitudes
        expanded_H = tf.tile(tf.expand_dims(check_matrix_H,axis=0),(vc_matrix.shape[0],1,1))
        reformed_H = tf.reshape(expanded_H,[-1,self.code.H.shape[1]])
        index_matrix = tf.where(reformed_H)
        nonzero_matrix =tf.reshape(tf.math.abs(vc_matrix)[expanded_H != 0],[-1,self.code.max_chk_degree])
        a_eye = tf.eye(self.code.max_chk_degree)
        ones = tf.ones(self.code.max_chk_degree)
        def expanded_row(i):
            test_data_t = tf.transpose(nonzero_matrix)
            mask_row = ones - a_eye[i]    
            left_matrix = tf.transpose(tf.boolean_mask(test_data_t,mask_row))  
            stacked_item = tf.sort(left_matrix,axis=-1,direction='ASCENDING')
            expanded_item = tf.expand_dims(stacked_item,axis=-1)
            list_row = self.check_model(expanded_item)
            return list_row
        list_d = list(map(expanded_row,range(self.code.max_chk_degree)))
        list_d = tf.concat(list_d,axis=1)
        cv_sparse_flattened = tf.squeeze(tf.reshape(list_d,[-1,1]))
        sp_input = tf.SparseTensor(dense_shape=reformed_H.shape,values=cv_sparse_flattened,indices = index_matrix)
        cv_matrix_dense = tf.sparse.to_dense(sp_input)
        cv_matrix = tf.stop_gradient(result_sign_matrix)*tf.reshape(cv_matrix_dense,shape=vc_matrix.shape)  
        updated_soft_cv_matrix_array = soft_cv_matrix_array.write(iteration, cv_matrix)   
        return updated_soft_cv_matrix_array
    
    def compute_cv_spa(self, soft_vc_matrix_array,soft_cv_matrix_array,iteration):
        vc_matrix = soft_vc_matrix_array.read(iteration)
        LLR_MAX = 10.0
        eps = 1e-5
        H = self.code.H
        # ===== sign =====
        vc_masked = tf.where(H == 1, vc_matrix, 1.0)
        signs = tf.where(vc_masked >= 0.0, 1.0, -1.0)
        total_sign = tf.reduce_prod(signs, axis=-1, keepdims=True)
        signs_i = total_sign * signs
        # ===== magnitude =====
        mags = tf.abs(vc_matrix)
        mags = tf.where(H == 1, mags, 1.0)       
        phi_all = self.phi(mags)
        phi_vals = tf.where(H == 1, phi_all, 0.0)
        sum_phi = tf.reduce_sum(phi_vals, axis=-1, keepdims=True)
        extr_phi = tf.maximum(sum_phi - phi_vals, eps)
        safe_extr = tf.where(H == 1, extr_phi, 1.0)
        mag_all = self.phi_inv(safe_extr)
        mag_i = tf.where(H == 1, mag_all, 0.0)
        cv_matrix = signs_i * mag_i
        cv_matrix = tf.clip_by_value(cv_matrix, -LLR_MAX, LLR_MAX)  
        updated_soft_cv_matrix_array = soft_cv_matrix_array.write(iteration, cv_matrix)  
        return updated_soft_cv_matrix_array
      
    def marginalize(self, super_input,soft_cv_matrix_array,soft_output_array,shifted_integers,iteration):
        cv_matrix = soft_cv_matrix_array.read(iteration)
        temp = tf.reduce_sum(cv_matrix,1)
        if GL.get_map('selected_decoder_type') in ['SPA-1','Check-S','QBP-S']:
            normalized_tensor = tf.nn.softplus(self.decoder_balance_normalizor)
        else:
            normalized_tensor = 1.
        #aligning with cycled input
        num_shifts = GL.get_map('num_shifts')
        if num_shifts > 0:
            batch_size = super_input.shape[0]//num_shifts
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
            external_sum = tensor1+interleaved+permutated
            normalized_tensor = tf.nn.softplus(self.decoder_balance_normalizor)
            soft_output = soft_output_array.read(iteration)+external_sum*normalized_tensor
        else:
            soft_output = soft_output_array.read(iteration)+temp
        return soft_output 
  
    @tf.function
    def phi(self, x, eps=1e-5):
        x = tf.maximum(x, eps)
        # φ(x) = log( (1+e^{-x}) / (1-e^{-x}) )
        exp_neg_x = tf.exp(-x)
        numerator = 1.0 + exp_neg_x
        denominator = 1.0 - exp_neg_x + eps
        return tf.math.log(numerator) - tf.math.log(denominator)

    @tf.function
    def phi_inv(self, y, eps=1e-10):
        y = tf.maximum(y, eps)
        exp_neg_y = tf.exp(-y)
        numerator = 1.0 + exp_neg_y
        denominator = 1.0 - exp_neg_y + eps
        return tf.math.log(numerator) - tf.math.log(denominator)
    
    def phi2(self, x, eps=1e-5):
        x = tf.maximum(x, eps)
        t = tf.exp(-x)                    
        return tf.math.log1p(t) - tf.math.log1p(-t + eps)
    def phi_inv2(self, y, eps=1e-5):
        y = tf.maximum(y, eps)
        t = tf.exp(-y)
        return tf.math.log1p(t) - tf.math.log1p(-t + eps)
    def get_evenly_shifted_integers(self,limit,num_count):
        start = np.random.randint(limit)
        step = limit//num_count
        sequence = (start + step * np.arange(num_count)) %limit      
        return list(sequence)
      
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
    
    def frobenius_automorphism(self,sequences, n):
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
        perm_matrix = tf.eye(n, dtype=sequences.dtype)
        perm_matrix = tf.gather(perm_matrix, positions, axis=1)
        
        # Apply the permutation
        permuted_sequences = tf.matmul(sequences, perm_matrix)
        
        return permuted_sequences
    
    def inverse_frobenius_automorphism(self,sequences, n):
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
        perm_matrix = tf.eye(n, dtype=sequences.dtype)
        perm_matrix = tf.gather(perm_matrix, inverse_positions, axis=1)
        
        # Apply the inverse permutation
        recovered_sequences = tf.matmul(sequences, perm_matrix)
        
        return recovered_sequences

    
class SPA_model(tf.keras.Model):
    def __init__(self,check_model=None,initial=None):
        super().__init__()
        self.decoder_layer = SPA_layer(check_model,initial)  # Explicitly track the layer
    def build(self, input_shape):
        # Convert TensorShape to plain Python dimensions for your layer
        if hasattr(input_shape, 'as_list'):
            processed_shape = input_shape.as_list()
        else:
            processed_shape = list(input_shape)       
        # Build your decoder layer with concrete dimensions
        self.decoder_layer.build(processed_shape)       
        # Skip super().build() entirely - it's often not needed
        self.built = True  # Manually mark as built
    def call(self,inputs): 
        output_list = self.decoder_layer(inputs)
        return output_list
   
    def get_eval(self,soft_output_array,labels):
        code = GL.get_map('code_parameters')
        num_iterations = GL.get_map('num_iterations')
        labels = tf.cast(labels,tf.int64)
        H = code.original_H
        soft_output = soft_output_array.read(num_iterations)
        tmp = tf.cast(tf.where(soft_output>0,0,1),tf.int64)
        syndrome = tf.matmul(tmp,H,transpose_b=True)%2
        index1 = np.nonzero(tf.reduce_sum(syndrome,-1))[0]
        err_batch = tf.where(tmp == labels,0,1)
        err_sum = tf.reduce_sum(err_batch,-1)
        BER_count = tf.reduce_sum(err_sum)
        FER_data = tf.where(err_sum!=0,1,0)
        FER_count = tf.math.count_nonzero(FER_data)
        #identify the indices of undected decoding errors        
        return FER_count, BER_count,index1   
  
    def collect_failed_input_output(self,soft_output_array,labels,indices):
        list_length = GL.get_map('num_iterations')+1
        soft_output_list = soft_output_array.stack()
        buffer_inputs = []
        buffer_labels = []
        #indices = tf.squeeze(index,1).numpy()
        for i in indices:
            for j in range(list_length):
                buffer_inputs.append(soft_output_list[j][i])    
                buffer_labels.append(labels[i])
        return buffer_inputs,buffer_labels  
    
class SPA_layer(tf.keras.layers.Layer):
    def __init__(self,check_model=None,initial=None ):
        super().__init__()
        self.decoder_type = GL.get_map('selected_decoder_type')
        self.num_iterations = GL.get_map('num_iterations')
        self.code = GL.get_map('code_parameters')
        self.H = self.code.H
        self.initial = initial if initial is not None else -2.
        self.supplement_matrix =  tf.expand_dims(tf.cast(1-self.H,dtype=tf.float32),0)
        self.check_model = check_model
        self.em = BPSK_EM_Estimator()
        self.indices = self.grab_index()
        self.decoder_balance_normalizor = self.add_weight(
            name='decoder_balance_normalizor',
            shape=[],
            trainable=True,
            initializer=tf.keras.initializers.Constant(self.initial),
            dtype=tf.float32)
    #V:vertical H:Horizontal D:dynamic S:Static  /  VSSL: Vertical Static/Dynamic Shared Layer
    def build(self, input_shape):   
        pass
    def call(self, inputs):
        noise_variance = GL.get_map('noise_variance')
        soft_input = 2/noise_variance*inputs[0]  #transformed int LLR               
        labels = inputs[1]
        outputs = self.belief_propagation_op(soft_input,labels)
        return outputs  # ✅ stack before returning
    def grab_index(self):
        n = self.code.max_chk_degree
        mask = 1.0 - tf.eye(n, dtype=tf.float32)
        positions = tf.where(tf.equal(mask, 1.0)) 
        indices = tf.reshape(positions[:, 1], [n, n-1]) 
        return indices
    def aggregate_cyclic_words(self,soft_input,labels):
        #cycling the input from start to end
        num_shifts = GL.get_map('num_shifts')
        if num_shifts > 0:
            soft_input_split = tf.concat([soft_input[:,0::2],soft_input[:,1::2]],axis=1)
            label_split = tf.concat([labels[:,0::2],labels[:,1::2]],axis=1)
            permutated_sequences = self.frobenius_automorphism(soft_input,soft_input.shape[1])
            permutated_labels = self.frobenius_automorphism(labels,labels.shape[1])            
            soft_input = tf.concat([soft_input,soft_input_split,permutated_sequences],0)  
            new_labels = tf.concat([labels,label_split,permutated_labels],0)  
            shifted_integers = self.get_evenly_shifted_integers(soft_input.shape[1],num_shifts)
            shifted_input_list = [tf.roll(soft_input,shifted_integers[i],axis=1)  for i in range(num_shifts)]
            shifted_label_list = [tf.roll(new_labels,shifted_integers[i],axis=1)  for i in range(num_shifts)]
            super_inputs = tf.concat(shifted_input_list,axis=0)
            super_labels = tf.concat(shifted_label_list,axis=0)
        else:
            super_inputs = soft_input
            super_labels = labels
            shifted_integers = [-1]
        return super_inputs,super_labels,shifted_integers                   
    def normalize_input(self,soft_input):
        mean = tf.reduce_mean(tf.abs(soft_input),axis=1,keepdims=True)
        base_line = 2/GL.get_map('noise_variance')
        scale_factor = base_line/mean
        soft_input = soft_input*scale_factor
        return soft_input             
    def belief_propagation_op(self, soft_input,labels):
        soft_output_array = tf.TensorArray(
            dtype=tf.float32,
            size=self.num_iterations+1,
            clear_after_read=False  # <-- Required if you want to read an index multiple times
        )
        # Write initial value
        iteration = 0
        loss = 0.
        soft_output_array = soft_output_array.write(iteration, soft_input)
        def condition(soft_output_array,labels,iteration,loss):
            return iteration < self.num_iterations
        def body(soft_output_array,labels,iteration,loss):
            soft_inputs = soft_output_array.read(iteration)
            super_inputs,super_labels,shifted_integers = self.aggregate_cyclic_words(soft_inputs,labels)
            vc_matrix = self.compute_vc(super_inputs)
            # compute cv
            #print(super_inputs)
            if self.decoder_type == 'SPA-1':
                cv_matrix = self.compute_cv_spa(vc_matrix)     
            if self.decoder_type == 'QBP-SF1':
                cv_matrix = self.compute_cv_sf1_reduced(vc_matrix)             
            if self.decoder_type == 'QBP-SF2':
                cv_matrix = self.compute_cv_sf2_reduced(vc_matrix)     
            if self.decoder_type == 'QBP-SF3':
                cv_matrix = self.compute_cv_sf3_reduced(vc_matrix)    
            # get output for this iteration
            soft_output = self.marginalize(super_inputs,cv_matrix,soft_output_array,shifted_integers,iteration)  
            iteration += 1 
            # rectify the mean of inputs
            soft_output = self.normalize_input(soft_output)           
            soft_output_array = soft_output_array.write(iteration, soft_output)      
            loss = self.calculation_loss(soft_output,labels,loss) 
            return soft_output_array,labels,iteration,loss
    
        soft_output_array,labels,iteration,loss = tf.while_loop(
            condition,
            body,
            loop_vars=[soft_output_array,labels,iteration,loss ]
        )   
        return soft_output_array,labels,loss
 
    def calculation_loss(self,soft_output,labels,loss):
         #cross entroy
        labels = tf.cast(labels,tf.float32)
        CE_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=-soft_output, labels=labels)) 
        return CE_loss+loss
    
    def compute_vc(self,super_input):
        check_matrix_H = tf.cast(self.code.H,tf.float32)   
        vc_matrix = tf.expand_dims(super_input,axis=1)*check_matrix_H           
        return vc_matrix               
    
    def compute_cv_sf1_reduced(self,vc_matrix):
        check_matrix_H = self.code.H
        #operands sign processing 
        supplement_matrix = tf.cast(1-check_matrix_H,dtype=tf.float32)
        supplement_matrix = tf.expand_dims(supplement_matrix,0)
        sign_info = supplement_matrix + vc_matrix
        vc_matrix_sign = tf.sign(sign_info)
        temp = tf.reduce_prod(vc_matrix_sign,axis=-1)
        temp = tf.expand_dims(temp,axis=-1)
        transition_sign_matrix = temp*check_matrix_H
        result_sign_matrix = transition_sign_matrix*vc_matrix_sign 
        #oprations on magnitudes
        expanded_H = tf.tile(tf.expand_dims(check_matrix_H,axis=0),(vc_matrix.shape[0],1,1))
        mask = tf.cast(expanded_H != 0, tf.bool)
        nonzero_indices = tf.where(mask)  # [total_nonzero, 3] - batch, row, col
        vc_magnitude = tf.math.abs(vc_matrix)
        nonzero_values = tf.boolean_mask(vc_magnitude, mask)
        batch_check_count = tf.shape(vc_matrix)[0] * tf.shape(vc_matrix)[1]
        nonzero_matrix = tf.reshape(nonzero_values, [batch_check_count, self.code.max_chk_degree])
        
        sorted_matrix = tf.sort(nonzero_matrix,axis=-1,direction='DESCENDING')
        sorted_indices = tf.argsort(nonzero_matrix, axis=-1, direction='DESCENDING')
        expanded_sorted_matrix = tf.expand_dims(sorted_matrix,axis=-1)
        updated_matrix = self.check_model(expanded_sorted_matrix,training=True)
        reverse_indices = tf.argsort(sorted_indices, axis=-1)
        recovered_unsorted = tf.gather(updated_matrix, reverse_indices, batch_dims=1)
        
        updates = tf.reshape(recovered_unsorted, [-1])  

        updates = updates[:tf.shape(nonzero_indices)[0]]

        cv_matrix_dense = tf.tensor_scatter_nd_update(vc_magnitude, nonzero_indices, updates)
        
        cv_matrix = tf.stop_gradient(result_sign_matrix)*cv_matrix_dense 
        return cv_matrix
   
    def compute_cv_sf2_reduced(self,vc_matrix):
        eps = 1e-20
        clip_val=0.9999
        check_matrix_H = self.code.H
        #operands sign processing 
        supplement_matrix = tf.cast(1-check_matrix_H,dtype=tf.float32)
        supplement_matrix = tf.expand_dims(supplement_matrix,0)
        sign_info = supplement_matrix + vc_matrix
        vc_matrix_sign = tf.sign(sign_info)
        temp = tf.reduce_prod(vc_matrix_sign,axis=-1)
        temp = tf.expand_dims(temp,axis=-1)
        transition_sign_matrix = temp*check_matrix_H
        result_sign_matrix = transition_sign_matrix*vc_matrix_sign 
        #oprations on magnitudes
        expanded_H = tf.tile(tf.expand_dims(check_matrix_H,axis=0),(vc_matrix.shape[0],1,1))
        mask = tf.cast(expanded_H != 0, tf.bool)
        nonzero_indices = tf.where(mask)  # [total_nonzero, 3] - batch, row, col
        vc_magnitude = tf.math.abs(vc_matrix)
        nonzero_values = tf.boolean_mask(vc_magnitude, mask)
        batch_check_count = tf.shape(vc_matrix)[0] * tf.shape(vc_matrix)[1]
        nonzero_matrix = tf.reshape(nonzero_values, [batch_check_count, self.code.max_chk_degree])
        
        sorted_matrix = tf.sort(nonzero_matrix,axis=-1,direction='DESCENDING')
        sorted_indices = tf.argsort(nonzero_matrix, axis=-1, direction='DESCENDING')
        expanded_sorted_matrix = tf.expand_dims(sorted_matrix,axis=-1)
        product_vector = self.check_model(expanded_sorted_matrix,training=True)
        x_clipped = tf.clip_by_value(product_vector/(sorted_matrix+eps), -clip_val, clip_val)
        updated_matrix = 2.*tf.math.atanh(x_clipped)
        reverse_indices = tf.argsort(sorted_indices, axis=-1)
        recovered_unsorted = tf.gather(updated_matrix, reverse_indices, batch_dims=1)
        
        updates = tf.reshape(recovered_unsorted, [-1])  

        updates = updates[:tf.shape(nonzero_indices)[0]]

        cv_matrix_dense = tf.tensor_scatter_nd_update(vc_magnitude, nonzero_indices, updates)
        
        cv_matrix = tf.stop_gradient(result_sign_matrix)*cv_matrix_dense 
        return cv_matrix
    
    def compute_cv_sf3_reduced(self,vc_matrix):
        check_matrix_H = self.code.H
        #operands sign processing 
        supplement_matrix = tf.cast(1-check_matrix_H,dtype=tf.float32)
        supplement_matrix = tf.expand_dims(supplement_matrix,0)
        sign_info = supplement_matrix + vc_matrix
        vc_matrix_sign = tf.sign(sign_info)
        temp = tf.reduce_prod(vc_matrix_sign,axis=-1)
        temp = tf.expand_dims(temp,axis=-1)
        transition_sign_matrix = temp*check_matrix_H
        result_sign_matrix = transition_sign_matrix*vc_matrix_sign 
        #oprations on magnitudes
        expanded_H = tf.tile(tf.expand_dims(check_matrix_H,axis=0),(vc_matrix.shape[0],1,1))
        mask = tf.cast(expanded_H != 0, tf.bool)
        nonzero_indices = tf.where(mask)  # [total_nonzero, 3] - batch, row, col
        vc_magnitude = tf.math.abs(vc_matrix)
        nonzero_values = tf.boolean_mask(vc_magnitude, mask)
        batch_check_count = tf.shape(vc_matrix)[0] * tf.shape(vc_matrix)[1]
        nonzero_matrix = tf.reshape(nonzero_values, [batch_check_count, self.code.max_chk_degree])
        
        sorted_matrix = tf.sort(nonzero_matrix,axis=-1,direction='DESCENDING')
        sorted_indices = tf.argsort(nonzero_matrix, axis=-1, direction='DESCENDING')
        batch_size = vc_matrix.shape[0]*vc_matrix.shape[1]
        indices_batch = tf.tile(tf.expand_dims(self.indices, 0), [batch_size, 1, 1])
        dilated_inputs = tf.gather(sorted_matrix, indices_batch, batch_dims=1)  # (batch, n, n-1)
        dilated_matrix = tf.reshape(dilated_inputs,[-1,dilated_inputs.shape[-1]])
        expanded_sorted_matrix = tf.expand_dims(dilated_matrix,axis=-1)
        updated_matrix = self.check_model(expanded_sorted_matrix,training=True)
        reverse_indices = tf.argsort(sorted_indices, axis=-1)
        recovered_unsorted = tf.gather(updated_matrix, reverse_indices, batch_dims=1)
        
        updates = tf.reshape(recovered_unsorted, [-1])  

        updates = updates[:tf.shape(nonzero_indices)[0]]

        cv_matrix_dense = tf.tensor_scatter_nd_update(vc_magnitude, nonzero_indices, updates)
        
        cv_matrix = tf.stop_gradient(result_sign_matrix)*cv_matrix_dense 
        return cv_matrix
 
    def compute_cv_spa(self,vc_matrix):
        LLR_MAX = 10.0
        eps = 1e-5
        H = self.code.H
        # ===== sign =====
        vc_masked = tf.where(H == 1, vc_matrix, 1.0)
        signs = tf.where(vc_masked >= 0.0, 1.0, -1.0)
        total_sign = tf.reduce_prod(signs, axis=-1, keepdims=True)
        signs_i = total_sign * signs
        # ===== magnitude =====
        mags = tf.abs(vc_matrix)
        mags = tf.where(H == 1, mags, 1.0)       
        phi_all = self.phi(mags)
        phi_vals = tf.where(H == 1, phi_all, 0.0)
        sum_phi = tf.reduce_sum(phi_vals, axis=-1, keepdims=True)
        extr_phi = tf.maximum(sum_phi - phi_vals, eps)
        safe_extr = tf.where(H == 1, extr_phi, 1.0)
        mag_all = self.phi_inv(safe_extr)
        mag_i = tf.where(H == 1, mag_all, 0.0)
        cv_matrix = signs_i * mag_i
        cv_matrix = tf.clip_by_value(cv_matrix, -LLR_MAX, LLR_MAX)  
        return cv_matrix
      
    def marginalize(self, super_input,cv_matrix,soft_output_array,shifted_integers,iteration):
        temp = tf.reduce_sum(cv_matrix,1)
        normalized_tensor = tf.nn.softplus(self.decoder_balance_normalizor)
        #aligning with cycled input
        num_shifts = GL.get_map('num_shifts')
        if num_shifts > 0:
            batch_size = super_input.shape[0]//num_shifts
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
            external_sum = tensor1+interleaved+permutated
            normalized_tensor = tf.nn.softplus(self.decoder_balance_normalizor)
            soft_output = soft_output_array.read(iteration)+external_sum*normalized_tensor
        else:
            soft_output = soft_output_array.read(iteration)+temp
        return soft_output 
  
    @tf.function
    def phi(self, x, eps=1e-5):
        x = tf.maximum(x, eps)
        # φ(x) = log( (1+e^{-x}) / (1-e^{-x}) )
        exp_neg_x = tf.exp(-x)
        numerator = 1.0 + exp_neg_x
        denominator = 1.0 - exp_neg_x + eps
        return tf.math.log(numerator) - tf.math.log(denominator)

    @tf.function
    def phi_inv(self, y, eps=1e-10):
        y = tf.maximum(y, eps)
        exp_neg_y = tf.exp(-y)
        numerator = 1.0 + exp_neg_y
        denominator = 1.0 - exp_neg_y + eps
        return tf.math.log(numerator) - tf.math.log(denominator)
    
    def phi2(self, x, eps=1e-5):
        x = tf.maximum(x, eps)
        t = tf.exp(-x)                    
        return tf.math.log1p(t) - tf.math.log1p(-t + eps)
    def phi_inv2(self, y, eps=1e-5):
        y = tf.maximum(y, eps)
        t = tf.exp(-y)
        return tf.math.log1p(t) - tf.math.log1p(-t + eps)
    def get_evenly_shifted_integers(self,limit,num_count):
        start = np.random.randint(limit)
        step = limit//num_count
        sequence = (start + step * np.arange(num_count)) %limit      
        return list(sequence)
      
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
    
    def frobenius_automorphism(self,sequences, n):
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
        perm_matrix = tf.eye(n, dtype=sequences.dtype)
        perm_matrix = tf.gather(perm_matrix, positions, axis=1)
        
        # Apply the permutation
        permuted_sequences = tf.matmul(sequences, perm_matrix)
        
        return permuted_sequences
    
    def inverse_frobenius_automorphism(self,sequences, n):
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
        perm_matrix = tf.eye(n, dtype=sequences.dtype)
        perm_matrix = tf.gather(perm_matrix, inverse_positions, axis=1)
        
        # Apply the inverse permutation
        recovered_sequences = tf.matmul(sequences, perm_matrix)
        
        return recovered_sequences
    
class NMS_model(tf.keras.Model):
    def __init__(self,SE_instance=None,initial=None):
        super().__init__()
        self.decoder_layer = Decoder_Layer(SE_instance,initial)  # Explicitly track the layer
    def build(self, input_shape):
        # Convert TensorShape to plain Python dimensions for your layer
        if hasattr(input_shape, 'as_list'):
            processed_shape = input_shape.as_list()
        else:
            processed_shape = list(input_shape)       
        # Build your decoder layer with concrete dimensions
        self.decoder_layer.build(processed_shape)       
        # Skip super().build() entirely - it's often not needed
        self.built = True  # Manually mark as built
    def call(self,inputs): 
        soft_output_list,label,loss = self.decoder_layer(inputs)
        return soft_output_list,label,loss
    
    def collect_failed_input_output(self,soft_output_list,labels,indices):
        num_iterations = GL.get_map('num_iterations')
        list_length = num_iterations + 1
        buffer_inputs = []
        buffer_labels = []
        #indices = tf.squeeze(index,1).numpy()
        for i in indices:
            for j in range(list_length):
                buffer_inputs.append(soft_output_list[j][i])    
                buffer_labels.append(labels[i])
        return buffer_inputs,buffer_labels     

    def get_eval_fer(self,soft_output_list,labels):
        soft_output = soft_output_list[-1]
        tmp = tf.cast(tf.where(soft_output>0,0,1),tf.int64)
        err_batch = tf.where(tmp == labels,0,1)
        err_sum = tf.reduce_sum(err_batch,-1)
        FER_data = tf.where(err_sum!=0,1,0)     
        FER_num = tf.math.count_nonzero(FER_data)
        #identify the indices of undected decoding errors        
        return FER_num.numpy() 

    def new_get_eval(self,Model_undetected,soft_output_list,labels):
        code = GL.get_map('code_parameters')
        num_iterations = GL.get_map('num_iterations')
        soft_margin = GL.get_map('soft_margin')
        H = code.original_H
        soft_output = soft_output_list[-1]
        final_hard_decision = tf.cast(tf.where(soft_output>0,0,1),tf.int64)
        syndrome = tf.matmul(final_hard_decision,H,transpose_b=True)%2
        nms_declare_n_index = np.where(np.sum(syndrome,-1)!=0)[0]
        nms_declare_p_index = np.where(np.sum(syndrome,-1)==0)[0]
        delta_matrix = self.transform_samples(soft_output_list)
        passed_list = [delta_matrix[i] for i in nms_declare_p_index]
        p_matrix = tf.reshape(passed_list,[-1,num_iterations,1])
        outputs = Model_undetected(p_matrix)
        #partition the outputs into two categories
        compact_index = np.where(outputs[:,0]-outputs[:,1]>soft_margin)[0]
        #passed index after model
        ude_declare_p_index = [nms_declare_p_index[i] for i in compact_index]
        #failed index after model
        ude_declare_n_index = set(nms_declare_p_index)-set(ude_declare_p_index)
        declare_n_index = list(ude_declare_n_index | set(nms_declare_n_index))
        #index_fail = list(set(index_fail))
        #ground truth comparison to determine FER and BER
        correct_counter = 0
        err_batch = tf.where(final_hard_decision == labels,0,1)
        err_word_indicator = tf.reduce_sum(err_batch,-1)
        err_bit_sum = tf.reduce_sum(err_word_indicator)
        false_p_index = []
        false_n_index = []
        for i in ude_declare_p_index:
            correct_sign = tf.where(err_word_indicator[i]==0,1,0)
            correct_counter += correct_sign 
            if not correct_sign:
                false_p_index.append(i)
        for i in ude_declare_n_index:
            correct_sign = tf.where(err_word_indicator[i]==0,1,0)
            if correct_sign:
                false_n_index.append(i)
        FER_count = soft_output.shape[0]- correct_counter              
        BER_count = err_bit_sum   
        return FER_count,BER_count,false_p_index,false_n_index,declare_n_index  
    
    def get_eval(self,soft_output_list,labels):
        code = GL.get_map('code_parameters')
        labels = tf.cast(labels,tf.int64)
        H = code.original_H
        soft_output = soft_output_list[-1]
        tmp = tf.cast(tf.where(soft_output>0,0,1),tf.int64)
        syndrome = tf.matmul(tmp,H,transpose_b=True)%2
        index1 = np.nonzero(tf.reduce_sum(syndrome,-1))[0]
        err_batch = tf.where(tmp == labels,0,1)
        err_sum = tf.reduce_sum(err_batch,-1)
        BER_count = tf.reduce_sum(err_sum)
        FER_data = tf.where(err_sum!=0,1,0)
        FER_count = tf.math.count_nonzero(FER_data)
        #identify the indices of undected decoding errors        
        return FER_count, BER_count,index1     
    
    def create_samples(self,soft_output_list, labels):
        print('.',end=' ')
        label_bool = tf.cast(labels, tf.bool)
        code = GL.get_map('code_parameters')
        H = code.original_H
        soft_output = soft_output_list[-1]         
        final_hard = tf.cast(tf.where(soft_output>0,0,1),tf.int64)
        delta_list = self.transform_samples(soft_output_list)
        syndrome = tf.matmul(final_hard,H,transpose_b=True)%2
        #indices of discarded pair
        index1 = np.nonzero(tf.reduce_sum(syndrome,-1))[0] 
        #cared indices to be classifed as positive or negative pair
        index2 = np.where(tf.reduce_sum(syndrome,-1) == 0)[0]
        pairs, pair_labels = [], []
        ground_labels = []
        for i in index2:
            output_hard_decision = tf.cast((soft_output[i] < 0),tf.bool)       
            err_indicator = tf.math.logical_xor(output_hard_decision, label_bool[i])
            Find_FER_sign = tf.reduce_any(err_indicator)
            pairs.append(delta_list[i])
            if Find_FER_sign:
                pair_labels.append(1)               
            else:
                pair_labels.append(0)
            ground_labels.append(labels[i])
        return tf.convert_to_tensor(pairs), tf.convert_to_tensor(pair_labels),tf.convert_to_tensor(ground_labels),index1  

    def transform_samples(self,soft_output_list):
        list_length = len(soft_output_list)
        initial_input = soft_output_list[0]
        final_output = soft_output_list[-1]         
        final_hard = tf.where(final_output>0,0,1)
        delta_list = []
        for i in range(list_length-1): 
            current_iteration_hard =  tf.where(soft_output_list[i]>0,0,1)
            difference_indicator = tf.cast((final_hard+current_iteration_hard)%2,tf.float32)
            differed_distance = tf.reduce_sum(difference_indicator*abs(initial_input),1,keepdims=True)
            delta_list.append(differed_distance)
        delta_matrix = tf.concat(delta_list,1)
        return delta_matrix
    
class Decoder_Layer(tf.keras.layers.Layer):
    def __init__(self,initial=None):
        super().__init__()
        self.decoder_type = GL.get_map('selected_decoder_type')
        self.num_iterations = GL.get_map('num_iterations')
        self.code = GL.get_map('code_parameters')       
        self.H = self.code.H
        self.initial = initial if initial is not None else -2.
        self.supplement_matrix =  tf.expand_dims(tf.cast(1-self.H,dtype=tf.float32),0)
        self.decoder_balance_normalizor = self.add_weight(
            name='decoder_balance_normalizor',
            shape=[],
            trainable=True,
            initializer=tf.keras.initializers.Constant(self.initial),
            dtype=tf.float32
        )               
    #V:vertical H:Horizontal D:dynamic S:Static  /  VSSL: Vertical Static/Dynamic Shared Layer
    def build(self, input_shape): 
        pass
    def call(self, inputs):
        # VERIFICATION: Ensure weight persists
        if not hasattr(self, 'decoder_balance_normalizor'):
            raise RuntimeError("Weight lost during execution!")
        soft_input = inputs[0]
        labels = inputs[1]    
        outputs = self.belief_propagation_op(soft_input, labels)
        soft_output_array, label, loss = outputs
        return soft_output_array.stack(), label, loss  # ✅ stack before returning
   
    def aggregate_cyclic_words(self,soft_input,labels):
        #cycling the input from start to end
        num_shifts = GL.get_map('num_shifts')
        if num_shifts > 0:
            soft_input_split = tf.concat([soft_input[:,0::2],soft_input[:,1::2]],axis=1)
            label_split = tf.concat([labels[:,0::2],labels[:,1::2]],axis=1)
            permutated_sequences = self.frobenius_automorphism(soft_input,soft_input.shape[1])
            permutated_labels = self.frobenius_automorphism(labels,labels.shape[1])            
            soft_input = tf.concat([soft_input,soft_input_split,permutated_sequences],0)  
            new_labels = tf.concat([labels,label_split,permutated_labels],0)  
            shifted_integers = self.get_evenly_shifted_integers(soft_input.shape[1],num_shifts)
            shifted_input_list = [tf.roll(soft_input,shifted_integers[i],axis=1)  for i in range(num_shifts)]
            shifted_label_list = [tf.roll(new_labels,shifted_integers[i],axis=1)  for i in range(num_shifts)]
            super_inputs = tf.concat(shifted_input_list,axis=0)
            super_labels = tf.concat(shifted_label_list,axis=0)
        else:
            super_inputs = soft_input
            super_labels = labels
            shifted_integers = [-1]
        return super_inputs,super_labels,shifted_integers                  

    def belief_propagation_op(self, soft_input, labels):
        soft_output_array = tf.TensorArray(
            dtype=tf.float32,
            size=self.num_iterations + 1,
            clear_after_read=False  # <-- Required if you want to read an index multiple times
        )
        # Write initial value
        soft_output_array = soft_output_array.write(0, soft_input)
    
        def condition(soft_output_array, labels,iteration,loss):
            return iteration < self.num_iterations
        def body(soft_output_array,labels, iteration,loss):
            super_input,super_labels,shifted_integers = self.aggregate_cyclic_words(soft_output_array.read(iteration),labels)
            vc_matrix = self.compute_vc(super_input)
            # compute cv           
            cv_matrix = self.compute_cv_nms(vc_matrix)      
            soft_output = self.marginalize(cv_matrix, super_input,shifted_integers,iteration,soft_output_array)  
            iteration += 1   
            normalized_soft_output = self.normalize_input(soft_output)
            soft_output_array = soft_output_array.write(iteration, normalized_soft_output)
            loss = self.calculation_loss(normalized_soft_output,labels,loss)
            
            return soft_output_array,labels, iteration,loss
    
        soft_output_array, labels, iteration, loss = tf.while_loop(
            condition,
            body,
            loop_vars=[soft_output_array, labels, 0, 0.]
        )   
        return soft_output_array, labels,loss
            
    # compute messages from variable nodes to check nodes

    def compute_vc(self, soft_input):
        check_matrix_H = tf.cast(self.code.H,tf.float32)   
        vc_matrix = tf.expand_dims(soft_input,axis=1)*check_matrix_H          
        return vc_matrix      
    def optimized_topk(self,matrix_3d, mask_2d):
        # Flatten and mask
        a, b, c = tf.shape(matrix_3d)[0], tf.shape(matrix_3d)[1], tf.shape(matrix_3d)[2]
        masked = matrix_3d * tf.tile(tf.expand_dims(mask_2d, 0), [a, 1, 1])      
        # Replace zeros with large value
        processed = tf.where(masked == 0, 1e10, masked)    
        # Reshape and find top-2
        flattened = tf.reshape(processed, [a * b, c])
        topk = -tf.math.top_k(-flattened, k=2)[0]   
        # Extract results
        smallest = tf.reshape(topk[:, 0], [a, b, 1])
        second_smallest = tf.reshape(topk[:, 1], [a, b, 1])
        # Apply your update rule
        updated = tf.where(
            matrix_3d == smallest,
            second_smallest,
            smallest
        ) * mask_2d        
        return updated        
       
    def compute_cv_nms(self,vc_matrix):
        check_matrix_H = self.H
        #operands sign processing 
        sign_info = self.supplement_matrix + vc_matrix
        vc_matrix_sign = tf.sign(sign_info)
        temp1 = tf.reduce_prod(vc_matrix_sign,2,keepdims=True)
        transition_sign_matrix = temp1*check_matrix_H
        result_sign_matrix = transition_sign_matrix*vc_matrix_sign 
        # Get the number of rows and columns in the matrix
        batches, rows, cols = vc_matrix.shape
        updated_matrix = self.optimized_topk(tf.abs(vc_matrix), tf.cast(check_matrix_H,tf.float32))     
        cv_matrix = updated_matrix*tf.stop_gradient(result_sign_matrix) 
        return cv_matrix
   
    def calculation_loss(self,soft_output,labels,loss):
         #cross entroy
        labels = tf.cast(labels,tf.float32)
        CE_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=-soft_output, labels=labels)) 
        return CE_loss+loss
    
    def normalize_input(self,soft_input):
        mean = tf.reduce_mean(tf.abs(soft_input),axis=1,keepdims=True)
        soft_input = soft_input/mean
        return soft_input    
    
    def marginalize(self,cv_matrix, soft_input,shifted_integers,iteration,soft_output_array):
        temp = tf.reduce_sum(cv_matrix,1) 
        normalized_tensor = tf.nn.softplus(self.decoder_balance_normalizor)
        #aligning with cycled input
        num_shifts = GL.get_map('num_shifts')
        if num_shifts > 0:
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
            soft_output = soft_output_array.read(iteration)+(tensor1+interleaved+permutated)*normalized_tensor
        else:
            soft_output = soft_output_array.read(iteration)+temp
        return soft_output
    
    def get_evenly_shifted_integers(self,limit,num_count):
        start = np.random.randint(limit)
        step = limit//num_count
        sequence = (start + step * np.arange(num_count)) %limit      
        return list(sequence)
      
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
    
    def frobenius_automorphism(self,sequences, n):
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
        perm_matrix = tf.eye(n, dtype=sequences.dtype)
        perm_matrix = tf.gather(perm_matrix, positions, axis=1)
        
        # Apply the permutation
        permuted_sequences = tf.matmul(sequences, perm_matrix)
        
        return permuted_sequences
    
    def inverse_frobenius_automorphism(self,sequences, n):
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
        perm_matrix = tf.eye(n, dtype=sequences.dtype)
        perm_matrix = tf.gather(perm_matrix, inverse_positions, axis=1)
        
        # Apply the inverse permutation
        recovered_sequences = tf.matmul(sequences, perm_matrix)
        
        return recovered_sequences
    
#save modified data for postprocessing
def save_balanced_decoded_data(buffer_inputs,buffer_labels,file_dir):
    #code = GL.get_map('code_parameters')
    stacked_buffer_info = tf.stack(buffer_inputs)
    stacked_buffer_label = tf.stack(buffer_labels)
    print("\nData for retraining  with %d cases to be stored " % stacked_buffer_info.shape[0])
    #balanced_features,balanced_labels = create_balanced_batches(features, labels, unit_batch_size)   
    #data = (balanced_features,balanced_labels)
    data = (stacked_buffer_info,stacked_buffer_label)
    Data_gen.make_tfrecord(data, out_filename=file_dir)    
    print("Data storing finished!")
#save modified data for postprocessing
def save_decoded_data(buffer_inputs,buffer_labels,full_file_path):
    #code = GL.get_map('code_parameters')
    stacked_buffer_info = tf.stack(buffer_inputs)
    stacked_buffer_label = tf.stack(buffer_labels)
    print("\nData for retraining  with %d cases to be stored " % stacked_buffer_info.shape[0])
    data = (stacked_buffer_info,stacked_buffer_label)
    Data_gen.make_tfrecord(data,out_filename=full_file_path)    
    print("Data storing finished!")

def create_balanced_batches(features, labels, batch_size):
    # Separate minority and majority classes
    true_labels = features[:,-1:].astype(np.int32)
    minority_indices = np.where(true_labels == 1)[0]
    majority_indices = np.where(true_labels== 0)[0] 
    # Separate minority and majority classes
    minority_tuples = (features[minority_indices], labels[minority_indices])
    majority_tuples = (features[majority_indices], labels[majority_indices])
    minority_class = tf.data.Dataset.from_tensor_slices((minority_tuples,true_labels[minority_indices]))
    majority_class = tf.data.Dataset.from_tensor_slices((majority_tuples,true_labels[majority_indices]))

    # Repeat the minority class to match the majority class size
    #minority_class = minority_class.repeat(count=math.ceil(len(majority_indices)/len(minority_indices)))
    weights=[0.1, 1.2]
    current_multiples = len(majority_indices)*weights[0]/(len(minority_indices)*weights[1])
    minority_class = minority_class.repeat(count=math.ceil(current_multiples))

    # Interleave the two datasets to create balanced batches
    balanced_dataset = tf.data.Dataset.sample_from_datasets([minority_class, majority_class], weights)
    balanced_dataset = balanced_dataset.shuffle(buffer_size=10000).batch(batch_size)
    input_list = list(balanced_dataset.as_numpy_iterator())
    num_counter = len(input_list)
    balanced_features = np.concatenate([input_list[i][0][0] for i in range(num_counter)])
    balanced_labels =  np.concatenate([input_list[i][0][1] for i in range(num_counter)])
    return balanced_features,balanced_labels


def create_balanced_batches3(features, labels, batch_size):
    # Separate minority and majority classes
    minority_indices = np.where(tf.cast(features[:,-1],tf.int32) == 0)[0]
    majority_indices = np.where(tf.cast(features[:,-1],tf.int32) == 1)[0] 

    # Create datasets for minority and majority classes
    minority_class = tf.data.Dataset.from_tensor_slices((features[minority_indices], labels[minority_indices]))
    majority_class = tf.data.Dataset.from_tensor_slices((features[majority_indices], labels[majority_indices]))
    # Repeat the minority class 
    minority_class = minority_class.repeat().take(len(majority_indices))
    # Interleave the two datasets with specified weights
    balanced_dataset = tf.data.Dataset.sample_from_datasets(
        [minority_class, majority_class], weights=[0.5, 0.5]
    )
    minority_count = 0
    majority_count = 0
    for batch_features, batch_labels in balanced_dataset:
        minority_count += tf.reduce_sum(tf.cast(features[:,-1] == 0, tf.int32)).numpy()
        majority_count += tf.reduce_sum(tf.cast(features[:,-1] == 1, tf.int32)).numpy()

    print(f"Minority class examples: {minority_count}")
    print(f"Majority class examples: {majority_count}")
    print(f"Minority/Majority ratio: {minority_count / majority_count:.2f}")

    # Convert the dataset to NumPy arrays
    input_list = list(balanced_dataset.as_numpy_iterator())
    num_counter = len(input_list)
    balanced_features = np.concatenate([input_list[i][0] for i in range(num_counter)])
    balanced_labels = np.concatenate([input_list[i][1] for i in range(num_counter)])

    return balanced_features, balanced_labels

def calculate_loss(inputs,labels):
    labels = tf.cast(labels,tf.float32)  
    #measure discprepancy via cross entropy metric which acts as the loss definition for deep learning per batch         
    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=-inputs, labels=labels))
    return  loss

#postprocessing after first stage training
def generate_parity_pass_samples(Model,iterator):
    #collecting erroneous decoding info
    buffer_inputs = []
    buffer_labels = []
    ground_labels_list = []
    #query of size of input feedings
    input_list = list(iterator.as_numpy_iterator())
    num_counter = len(input_list) 
    for i in range(num_counter):
        if not (i+1) % 100:
            print("Total ",i+1," batches are processed!")
        inputs = input_list[i]
        soft_output_list,label,_ = Model(inputs)
        pair,pair_label,ground_labels,_ = Model.create_samples(soft_output_list,label)
        buffer_inputs.append( pair)
        buffer_labels.append(pair_label)
        ground_labels_list.append(ground_labels)
    sample_matrix = tf.concat(buffer_inputs,axis=0)
    ground_labels_matrix = tf.concat(ground_labels_list,axis=0)
    labels_vector =tf.reshape(tf.cast(tf.concat(buffer_labels,axis=0),tf.float32),[-1,1])
    feature_matrix = tf.concat([sample_matrix,labels_vector],1)
    return feature_matrix,ground_labels_matrix


#postprocessing after first stage training
def postprocess_training(NN_model, iterator):
    code = GL.get_map('code_parameters')
    #collecting erroneous decoding info
    buffer_inputs = []
    buffer_labels = []
    #query of size of input feedings
    input_list = list(iterator.as_numpy_iterator())
    num_counter = len(input_list) 
    FER_sum = 0
    BER_sum = 0
    num_samples = 0
    for i in range(num_counter):
        print(f'\r{i+1}-th batch: Processing...', end='', flush=True)
        if (i+1) % 20 == 0:
            print("Total ",i+1," batches are processed!")
            print(f'FER:{FER_sum/num_samples:.4f} BER:{BER_sum/(num_samples*code.n):.4f}')
        inputs = input_list[i]
        outputs_list,labels,_ = NN_model(inputs)
        FER_count,BER_count,delare_n_index = NN_model.get_eval(outputs_list,labels)     
        buffer_inputs_tmp,buffer_labels_tmp = NN_model.collect_failed_input_output(outputs_list,labels,delare_n_index)   
        buffer_inputs.append(buffer_inputs_tmp)
        buffer_labels.append(buffer_labels_tmp)
        num_samples += labels.shape[0]
        FER_sum += FER_count
        BER_sum += BER_count
        if FER_sum > GL.get_map('termination_threshold'):
            break
    buffer_inputs = [j for i in buffer_inputs for j in i]
    buffer_labels = [j for i in buffer_labels for j in i]
    return buffer_inputs,buffer_labels,FER_sum/num_samples,BER_sum/(num_samples*code.n),num_samples

def grab_index():
    code = GL.get_map('code_parameters')
    n = code.max_chk_degree
    mask = 1.0 - tf.eye(n, dtype=tf.float32)
    positions = tf.where(tf.equal(mask, 1.0)) 
    indices = tf.reshape(positions[:, 1], [n, n-1]) 
    return indices
#main training process
def pre_process_inputs(current_decoder,inputs,nonzero_positions=''):         
    if current_decoder == 'Check-SF1': 
        input_data = tf.abs(inputs[0])
        labels = tf.abs(inputs[1])    
        extended_inputs = tf.expand_dims(input_data,-1)            
    if current_decoder == 'Check-SF2': 
        input_data = tf.abs(inputs[0])
        labels = tf.reduce_prod(tf.tanh(input_data/2),axis=-1,keepdims=True)  
        extended_inputs = tf.expand_dims(input_data,-1) 
    if current_decoder == 'Check-SF3':  
        soft_inputs = tf.abs(inputs[0])
        batch_size = soft_inputs.shape[0]
        indices_batch = tf.tile(tf.expand_dims(nonzero_positions, 0), [batch_size, 1, 1])
        dilated_matrix= tf.gather(soft_inputs, indices_batch, batch_dims=1)  # (batch, n, n-1)
        last_dim = dilated_matrix.shape[-1]
        dilated_inputs = tf.reshape(dilated_matrix,[-1,last_dim])
        extended_inputs = tf.expand_dims(dilated_inputs,-1)                    
        labels = tf.abs(inputs[1]) 
    return extended_inputs,labels

def training_block_check_approx(Model,current_decoder,train_info):
    batch_index,exponential_decay, optimizer, selected_ds, log_info = train_info
    print_interval = GL.get_map('print_interval')
    batch_size = GL.get_map('unit_batch_size')
    termination_step = GL.get_map('iterate_termination_step')
    nonzero_positions = grab_index()
    summary_writer, manager_current = log_info
    ds_iter = iter(selected_ds)
    while True:
        if batch_index >= termination_step:
            break
        try:
            inputs = next(ds_iter)
            extended_inputs,labels = pre_process_inputs(current_decoder,inputs,nonzero_positions=nonzero_positions)
        except StopIteration:
            # In case dataset is finite and exhausted
            print("⚠️ Dataset exhausted. Consider adding `.repeat()` in dataset pipeline.")
            break
        with tf.GradientTape() as tape:          
            soft_outputs = Model(extended_inputs,training=True)
            loss = Model.loss_query(soft_outputs,labels,extended_inputs)
        grads = tape.gradient(loss, Model.trainable_variables)
        grads_and_vars = [(grad, var)
                          for grad, var in zip(grads, Model.trainable_variables)
                          if grad is not None]
        optimizer.apply_gradients(grads_and_vars)     
        batch_index += 1
        with summary_writer.as_default():
            tf.summary.scalar("loss", loss, step=batch_index)
        if batch_index % print_interval == 0:
            manager_current.save(checkpoint_number=batch_index)
            tf.print(f'S:{batch_index} Var:{Model.trainable_variables[-1].numpy():.3f} Lr:{exponential_decay(batch_index):.5f} Ls:{loss*batch_size:.3f}')
    return Model      
       
#main training process 
def training_block(NN_model, train_info):
    [batch_index, _, optimizer, train_iterator, \
                  logger_info] = train_info 
    
    print_interval = GL.get_map('print_interval')
    termination_step = GL.get_map('iterate_termination_step')
    accumulator = GradientAccumulator(accumulation_steps=1)
    summary_writer, manager_current = logger_info
                  
    ds_iter = iter(train_iterator)
    
    # Cache variables
    all_vars = NN_model.trainable_variables        
    while True:
        if batch_index >= termination_step:
            break
        try:
            inputs = next(ds_iter)
        except StopIteration:
            print("⚠️ Dataset exhausted. Consider adding `.repeat()` in dataset pipeline.")
            break

        with tf.GradientTape() as tape:
            outputs = NN_model(inputs)           
            soft_output_list, labels, loss = outputs
            scaled_loss = loss / accumulator.accumulation_steps   
         
        fer, ber, _ = NN_model.get_eval(soft_output_list, labels)          
        grads = tape.gradient(scaled_loss, all_vars)         
        # Accumulate and apply
        should_apply = accumulator.accumulate(grads, all_vars)
        if should_apply:
            grads_and_vars = accumulator.get_gradients_and_reset()
            if grads_and_vars:  
                capped_gradients = [(grad, var) 
                                   for grad, var in grads_and_vars if grad is not None]
                if capped_gradients:
                    optimizer.apply_gradients(capped_gradients)
        
        batch_index += 1 
        
        # Logging
        with summary_writer.as_default():
            tf.summary.scalar("loss", loss, step=batch_index)
            tf.summary.scalar("FER", fer/labels.shape[0], step=batch_index)
            tf.summary.scalar("BER", ber/(labels.shape[0]*labels.shape[1]), step=batch_index)
        if batch_index % print_interval == 0:
            manager_current.save(checkpoint_number=batch_index)
            loss_val = loss.numpy()
            fer_val = (fer / labels.shape[0]).numpy()
            ber_val = (ber / (labels.shape[0] * labels.shape[1])).numpy()
            print(f'Step:{batch_index:4d}|Var:{all_vars[-1].numpy():.3f}|Loss:{loss_val:.1f}|'
                  f'FER:{fer_val:.3f}|BER:{ber_val:.3f}')               
    return NN_model

class GradientAccumulator:
    def __init__(self, accumulation_steps=1):
        self.accumulation_steps = accumulation_steps
        self.accumulation_counter = 0
        self.variables = None
        self.accumulated_gradients = None
    
    def accumulate(self, gradients, variables):
        """Accumulate gradients from current batch."""
        # Initialize on first call
        if self.accumulation_counter == 0:
            self.variables = variables
            self.accumulated_gradients = [
                tf.Variable(tf.zeros_like(var), trainable=False, dtype=var.dtype)
                for var in variables
            ]
        
        # Accumulate gradients
        for i, (grad, var) in enumerate(zip(gradients, variables)):
            if grad is not None:
                clipped_grad = tf.clip_by_norm(grad, 0.1)
                self.accumulated_gradients[i].assign_add(clipped_grad)
        
        self.accumulation_counter += 1
        return self.accumulation_counter >= self.accumulation_steps
    
    def get_gradients_and_reset(self):
        """Get accumulated gradients and reset the accumulator."""
        if self.accumulated_gradients is None:
            return []
        
        # Prepare gradients and variables for optimizer (only non-zero)
        grads_and_vars = []
        for grad_var, orig_var in zip(self.accumulated_gradients, self.variables):
            # Check if gradient is non-zero
            if grad_var is not None:
                # Use tf.reduce_any to check if any element is non-zero
                if tf.reduce_any(tf.not_equal(grad_var, 0)):
                    grads_and_vars.append((grad_var, orig_var))
        
        # Reset
        self.accumulation_counter = 0
        self.variables = None
        self.accumulated_gradients = None
        
        return grads_and_vars
