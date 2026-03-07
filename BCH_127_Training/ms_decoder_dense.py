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
import scatter_exit as SE

def ebno_to_noise_variance(ebno_db, rate=1.0):
    # Convert dB to linear scale
    ebno_linear = 10**(ebno_db / 10.0)
    
    # Account for coding rate: E_s/N_0 = R × E_b/N_0
    esno_linear = rate * ebno_linear
    
    # Calculate noise variance
    # σ² = N_0/2 = 1/(2 × E_s/N_0)
    # Note: Signal power is normalized to 1 (E_s = 1)
    noise_variance = 1.0 / (2.0 * esno_linear)
    
    return noise_variance

def noise_variance_to_ebno(noise_variance, rate=1.0, is_complex=True):
    # Recover E_s/N_0 from noise variance
    # E_s/N_0 = 1/(2 × σ²)
    esno_linear = 1.0 / (2.0 * noise_variance)
    
    # Recover E_b/N_0 from E_s/N_0 and coding rate
    # E_b/N_0 = (E_s/N_0) / rate
    ebno_linear = esno_linear / rate
    
    # Convert to dB
    ebno_db = 10 * np.log10(ebno_linear)
    
    return ebno_db

def simulate_awgn_channel(symbols, ebno_db, rate=1.0):
    # Check if symbols are complex
    is_complex = np.iscomplexobj(symbols)
    
    # Calculate theoretical noise variance
    noise_variance = ebno_to_noise_variance(ebno_db, rate, is_complex)
    
    # Generate AWGN noise
    if is_complex:
        # Complex noise: independent real and imaginary components
        # Each component has variance σ²/2
        noise_real = np.random.normal(0, np.sqrt(noise_variance/2), symbols.shape)
        noise_imag = np.random.normal(0, np.sqrt(noise_variance/2), symbols.shape)
        noise = noise_real + 1j * noise_imag
    else:
        # Real noise
        noise = np.random.normal(0, np.sqrt(noise_variance), symbols.shape)
    
    # Add noise to symbols
    received_symbols = symbols + noise
    
    return received_symbols, noise_variance

def calculate_actual_snr(received_symbols, transmitted_symbols):
    # Calculate signal power (assume zero mean)
    signal_power = np.mean(np.abs(transmitted_symbols)**2)
    
    # Calculate noise power (variance of noise)
    noise = received_symbols - transmitted_symbols
    noise_variance = np.mean(np.abs(noise)**2)
    
    # Calculate SNR in dB
    snr_linear = signal_power / noise_variance
    snr_db = 10 * np.log10(snr_linear)
    
    return snr_db, noise_variance

#from distfit import distfit    
#np.random.seed(0)
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
        
        # Prepare gradients and variables for optimizer
        grads_and_vars = []
        for grad_var, orig_var in zip(self.accumulated_gradients, self.variables):
            grads_and_vars.append((grad_var, orig_var))
        
        # Reset
        self.accumulation_counter = 0
        self.variables = None
        self.accumulated_gradients = None
        
        return grads_and_vars
    
class Decoding_model(tf.keras.Model):
    def __init__(self,SE_instance):
        super().__init__()
        self.decoder_layer = Decoder_Layer(SE_instance)  # Explicitly track the layer
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
    def __init__(self,SE_instance,initial_value = -1.6):
        super().__init__()
        self.decoder_type = GL.get_map('selected_decoder_type')
        self.num_iterations = GL.get_map('num_iterations')
        self.code = GL.get_map('code_parameters')
        self.probe_MI = GL.get_map('probe_MI')
        self.SPA_indicator = GL.get_map('selected_decoder_type')=='SPA'
        self.H = self.code.H
        self.initials = initial_value
        self.supplement_matrix =  tf.expand_dims(tf.cast(1-self.H,dtype=tf.float32),0)
        self.SE = SE_instance
        # SOLUTION: Force immediate weight creation with explicit name
        with tf.init_scope():  # <-- CRITICAL FIX
            if self.decoder_type == 'NMS-1':
                self.decoder_check_normalizor = self.add_weight(
                    name='decoder_check_normalizor',
                    shape=[1],
                    trainable=True,
                    initializer=tf.keras.initializers.Constant(initial_value),
                    dtype=tf.float32
                )
            if self.decoder_type == 'SPA':
                self.decoder_balance_normalizor = self.add_weight(
                    name='decoder_balance_normalizor',
                    shape=[1],
                    trainable=True,
                    initializer=tf.keras.initializers.Constant(initial_value),
                    dtype=tf.float32
                )                
        self.built = True
    #V:vertical H:Horizontal D:dynamic S:Static  /  VSSL: Vertical Static/Dynamic Shared Layer
    def build(self, input_shape): 
        self.SE.clear() 
    def call(self, inputs):
        # VERIFICATION: Ensure weight persists
        if not hasattr(self, 'decoder_check_normalizor') and not hasattr(self, 'decoder_balance_normalizor'):
            raise RuntimeError("Weight lost during execution!")
        soft_input = inputs[0]
        if self.SPA_indicator:
            ebno_db = GL.get_map('snr_lo')
            rate = self.code.k/self.code.n
            noise_variance = ebno_to_noise_variance(ebno_db,rate)
            soft_input = 2/noise_variance*soft_input  #transformed int LLR 

        labels = inputs[1]    
        outputs = self.belief_propagation_op(soft_input, labels)
        soft_output_array, label, loss = outputs
        return soft_output_array.stack(), label, loss  # ✅ stack before returning
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
    
        def condition(soft_output_array, loss, labels, iteration):
            return iteration < self.num_iterations
        def body(soft_output_array,loss,labels, iteration):
            super_input,super_labels,shifted_integers = self.aggregate_cyclic_words(soft_output_array.read(iteration),labels)
            vc_matrix = self.compute_vc(super_input)
            # compute cv           
            if self.SPA_indicator:
                cv_matrix = self.compute_cv_spa(vc_matrix)
            else:
                cv_matrix = self.compute_cv_nms(vc_matrix)      
            if self.probe_MI:
                self.SE.record_vn(vc_matrix,super_labels)
                self.SE.record_cn(cv_matrix,super_labels)   
            # if iteration == self.num_iterations-1:
            #     self.SE.record_vn(vc_matrix,super_labels)
            #     self.SE.record_cn(cv_matrix,super_labels)  
            #     # get output for this iteration
            #     loss = self.calculation_loss(-1,loss)
            soft_output_array = self.marginalize(cv_matrix, super_input,shifted_integers,iteration,soft_output_array)  
            iteration += 1   
            soft_output = soft_output_array.read(iteration)
            loss = self.calculation_loss(soft_output,labels,loss)
            
            return soft_output_array,loss,labels, iteration
    
        soft_output_array, loss, labels, iteration = tf.while_loop(
            condition,
            body,
            loop_vars=[soft_output_array, 0., labels, 0]
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
        if GL.get_map('selected_decoder_type') in ['NMS-1']:
          normalized_tensor = tf.nn.softplus(self.decoder_check_normalizor)
        cv_matrix = normalized_tensor*updated_matrix*tf.stop_gradient(result_sign_matrix) 
        return cv_matrix
    def compute_cv_spa1(self,vc_matrix):
        vc_matrix = tf.clip_by_value(vc_matrix, -15, 15)
        NaN_count = tf.reduce_sum(tf.cast(tf.math.is_nan(vc_matrix), tf.int32))
        if NaN_count > 0:
            print(f"NaN count: {NaN_count}")

        vc_matrix = tf.tanh(vc_matrix / 2.0) #tanh function applied 
        supple_matrix = 1 - self.code.H
        vc_matrix = vc_matrix+supple_matrix
        vc_matrix = tf.where(abs(vc_matrix)>0,vc_matrix,1e-10)
        temp = tf.reduce_prod(vc_matrix,2)                        
        temp = tf.expand_dims(temp,2)
        temp = temp*self.code.H
        cv_matrix = temp / vc_matrix
        cv_matrix = 2*tf.math.atanh(cv_matrix)         
        return cv_matrix    

    @tf.function
    def phi(self, x, eps=1e-5):
        x = tf.maximum(x, eps)
        t = tf.exp(-x)                    
        return tf.math.log1p(t) - tf.math.log1p(-t + eps)

    @tf.function
    def phi_inv(self, y, eps=1e-5):
        y = tf.maximum(y, eps)
        t = tf.exp(-y)
        return tf.math.log1p(t) - tf.math.log1p(-t + eps)
        
    def compute_cv_spa(self, vc_matrix):
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
    def calculation_loss(self,soft_output,labels,loss):
         #cross entroy
        labels = tf.cast(labels,tf.float32)
        CE_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=-soft_output, labels=labels)) 
        return CE_loss+loss
    def calculation_loss2(self,iteration,loss):
         #cross entroy
        sum_CHA = 1-self.SE.IA[iteration][2]
        sum_CHE = 1-self.SE.IE[iteration][2]
        coefficient = tf.where(sum_CHA<sum_CHE,1.0,1.0)
        CE_loss = coefficient*(sum_CHA+sum_CHE)
        return CE_loss
    def marginalize(self,cv_matrix, soft_input,shifted_integers,iteration,soft_output_array):
        temp = tf.reduce_sum(cv_matrix,1)
        if GL.get_map('selected_decoder_type') in ['SPA']:
            normalized_tensor = tf.nn.softplus(self.decoder_balance_normalizor)
        else:
            normalized_tensor = 1.
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
        soft_output_array = soft_output_array.write(iteration + 1, soft_output)
        return soft_output_array 
    #combine messages to get posterior LLRs

def retore_saved_model(restore_ckpts_dir,restore_step,ckpt_nm):
    print("Ready to restore a saved latest or designated model!")
    ckpt = tf.train.get_checkpoint_state(restore_ckpts_dir)
    if ckpt and ckpt.model_checkpoint_path: # ckpt.model_checkpoint_path means the latest ckpt
      if restore_step == 'latest':
        ckpt_f = tf.train.latest_checkpoint(restore_ckpts_dir)
        start_step = int(ckpt_f.split('-')[-1]) + 1
      else:
        ckpt_f = restore_ckpts_dir+ckpt_nm+'-'+restore_step
        start_step = int(restore_step)+1
      print('Loading wgt file: '+ ckpt_f)   
    else:
      print('Error, no qualified file found')
    return start_step,ckpt_f
#save modified data for postprocessing
def save_balanced_decoded_data(buffer_inputs,buffer_labels,file_dir):
    unit_batch_size = GL.get_map('unit_batch_size')
    #code = GL.get_map('code_parameters')
    stacked_buffer_info = tf.stack(buffer_inputs)
    stacked_buffer_label = tf.stack(buffer_labels)
    print("\nData for retraining  with %d cases to be stored " % stacked_buffer_info.shape[0])
    (features, labels) = (stacked_buffer_info.numpy(),stacked_buffer_label.numpy())
    #balanced_features,balanced_labels = create_balanced_batches(features, labels, unit_batch_size)   
    #data = (balanced_features,balanced_labels)
    data = (features,labels)
    Data_gen.make_tfrecord(data, out_filename=file_dir)    
    print("Data storing finished!")
#save modified data for postprocessing
def save_decoded_data(buffer_inputs,buffer_labels,file_dir):
    #code = GL.get_map('code_parameters')
    stacked_buffer_info = tf.stack(buffer_inputs)
    stacked_buffer_label = tf.stack(buffer_labels)
    print("\nData for retraining  with %d cases to be stored " % stacked_buffer_info.shape[0])
    (features, labels) = (stacked_buffer_info.numpy(),stacked_buffer_label.numpy()) 
    data = (features,labels)
    Data_gen.make_tfrecord(data,out_filename=file_dir)    
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

    # Shuffle and batch the dataset
    #balanced_dataset = balanced_dataset.shuffle(buffer_size=10000).batch(batch_size)

    # Debug: Count the number of examples from each class in the dataset
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
def postprocess_training(Model_NMS, iterator):
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
        if (i+1) % 100 == 0:
            print("Total ",i+1," batches are processed!")
            print(f'FER:{FER_sum/num_samples:.4f} BER:{BER_sum/(num_samples*code.n):.4f}')
        inputs = input_list[i]
        soft_output_list,label,_ = Model_NMS(inputs)
        FER_count,BER_count,delare_n_index = Model_NMS.get_eval(soft_output_list,label)     
        buffer_inputs_tmp,buffer_labels_tmp = Model_NMS.collect_failed_input_output(soft_output_list,label,delare_n_index)   
        buffer_inputs.append(buffer_inputs_tmp)
        buffer_labels.append(buffer_labels_tmp)
        num_samples += label.shape[0]
        FER_sum += FER_count
        BER_sum += BER_count
    buffer_inputs = [j for i in buffer_inputs for j in i]
    buffer_labels = [j for i in buffer_labels for j in i]
    return buffer_inputs,buffer_labels,FER_sum/num_samples,BER_sum/(num_samples*code.n),num_samples
    
#main training process
def training_undetected_block(batch_index,Model,optimizer,exponential_decay,selected_ds,log_info,restore_info):
    #query of size of input feedings
    print_interval = GL.get_map('print_interval')
    record_interval = GL.get_map('record_interval')
    termination_step = GL.get_map('ude_termination_step')
    ds_iter = iter(selected_ds.repeat())
    summary_writer,manager_current = log_info
    _,_,ckpts_dir_par,_= restore_info
    counter_sum = [0,0,0]   
    while True:
        if batch_index >= termination_step:
            break
        try:
            origin_inputs = next(ds_iter)
        except StopIteration:
            # In case dataset is finite and exhausted
            print("⚠️ Dataset exhausted. Consider adding `.repeat()` in dataset pipeline.")
            break
        with tf.GradientTape() as tape:
            inputs,labels = Model.pre_process_inputs(origin_inputs)
            outputs = Model(inputs)
            loss,counters =Model.calculate_loss(inputs,outputs,labels)
        grads = tape.gradient(loss,Model.variables)
        counter_sum[0] += counters[0]  #true positives and true negatives
        counter_sum[1] += counters[1]  #false negatives
        counter_sum[2] += counters[2]  #false positives
        grads_and_vars=zip(grads, Model.variables)
        capped_gradients = [(tf.clip_by_norm(grad,1e2), var) for grad, var in grads_and_vars if grad is not None]
        #capped_gradients = [(tf.clip_by_value(grad,-1,1), var) for grad, var in grads_and_vars if grad is not None]
        optimizer.apply_gradients(capped_gradients)
        batch_index += 1
        with summary_writer.as_default():                               # the logger to be used
            tf.summary.scalar("loss", loss, step=batch_index)
        # log to stdout 
        if batch_index % print_interval== 0: 
            manager_current.save(checkpoint_number=batch_index)
            tf.print(f"Step:{batch_index:4d} Lr:{exponential_decay(batch_index):.3f} Ls:{loss:.3f} Counters:{np.array(counter_sum)/np.sum(counter_sum)}") 
        if batch_index % record_interval == 0:
            print("For all layers at the %4d-th step:"%batch_index)
            manager_current.save(checkpoint_number=batch_index)
            # for variable in Model.variables:
            #     print(str(variable.numpy()))  
            with open(ckpts_dir_par+'values.txt','a+') as f:
                f.write("For all layers at the %4d-th step:\n"%batch_index)
                for variable in Model.variables:
                    f.write(variable.name+' '+str(variable.numpy())) 
                f.write('\n')  
    origin_inputs = next(ds_iter)
    inputs,labels = Model.pre_process_inputs(origin_inputs)
    _ = Model(inputs) #in case of loading model from file, to activate model.
    print("Final selected parameters:")
    for weight in  Model.get_weights():
      print(weight)
    return Model  

    
#main training process
def training_block(batch_index, Model, optimizer, exponential_decay, selected_ds, log_info, restore_info):
    print_interval = GL.get_map('print_interval')
    record_interval = GL.get_map('record_interval')
    termination_step = GL.get_map('nms_termination_step')
    accumulator = GradientAccumulator(accumulation_steps=3)
    summary_writer, manager_current = log_info
    ckpts_dir, ckpt_nm, ckpts_dir_par, restore_step = restore_info
    ds_iter = iter(selected_ds)
    while True:
        if batch_index >= termination_step:
            break
        try:
            inputs = next(ds_iter)
        except StopIteration:
            # In case dataset is finite and exhausted
            print("⚠️ Dataset exhausted. Consider adding `.repeat()` in dataset pipeline.")
            break
        with tf.GradientTape() as tape:
            soft_output_list, label, loss = Model(inputs)
            # scale loss
            scaled_loss = loss / accumulator.accumulation_steps
            grads = tape.gradient(scaled_loss, Model.trainable_variables)  
        fer, ber, _ = Model.get_eval(soft_output_list, label)
        # accumulate gradients
        should_apply = accumulator.accumulate(grads, Model.trainable_variables)   
        # Apply gradients once condition is satisfied
        if should_apply:
            grads_and_vars = accumulator.get_gradients_and_reset()
            if grads_and_vars:
                capped_gradients = [(tf.clip_by_norm(grad,1e3), var) for grad, var in grads_and_vars if grad is not None]
                optimizer.apply_gradients(capped_gradients)      
        batch_index += 1
        with summary_writer.as_default():
            tf.summary.scalar("loss", loss, step=batch_index)
            tf.summary.scalar("FER", fer/label.shape[0], step=batch_index)
            tf.summary.scalar("BER", ber/(label.shape[0]*label.shape[1]), step=batch_index)
        if batch_index % print_interval == 0:
            manager_current.save(checkpoint_number=batch_index)
            print("Parameter evaluation:")
            for weight in  Model.get_weights():
              print(weight)
            tf.print("Step%4d: Lr:%.3f Ls:%.4f FER:%.3f BER:%.4f"%\
            (batch_index,exponential_decay(batch_index),loss, fer/label.shape[0], ber/(label.shape[0]*label.shape[1]) )) 
        if batch_index % record_interval == 0:
            with open(ckpts_dir_par + 'values.txt', 'a+') as f:
                f.write("For all layers at %4d-th step:\n" % batch_index)
                for variable in Model.variables:
                    f.write(variable.name + ' ' + str(variable.numpy()) + '\n') 
    inputs = next(ds_iter)
    _ = Model(inputs)        #in case of loading model from file, to activate model.
    print("Final selected parameters:")
    for weight in  Model.get_weights():
      print(weight)
    return Model          