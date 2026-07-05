# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 10:47:08 2022

@author: Administrator
"""
import tensorflow as tf
import globalmap as GL

class Decoding_model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layer = Decoder_Layer()
    def call(self,inputs,Lth_layer,scale_factor): 
        bp_result = self.layer(inputs,Lth_layer,scale_factor)
        return bp_result
       
class Decoder_Layer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.code = GL.get_map('code_parameters')
        self.unit_batch_size = GL.get_map('unit_batch_size')
 # Code for model call (handles inputs and returns outputs)
    def call(self,inputs,Lth_layer,scale_factor):
        self.layer_iterations = scale_factor*(Lth_layer+1)
        soft_input = inputs[0]
        labels = inputs[1]     
        bp_result = self.belief_propagation_op(soft_input,labels,scale_factor)  
        return bp_result
                
# builds a belief propagation TF graph
    def belief_propagation_op(self,soft_input,labels,scale_factor):
        global soft_output_list 
        
        soft_output_list= []
        sigma = GL.get_map(('noise_standard_variance'))
        soft_input = 2*soft_input/(sigma**2)

        tf.while_loop(
            self.continue_condition, # iteration < max iteration?
            self.belief_propagation_iteration, # compute messages for this iteration
            loop_vars = [
                soft_input, # soft input for this iteration
                labels,
                0, # iteration number
                tf.zeros([self.unit_batch_size,self.code.check_matrix_row,self.code.check_matrix_column],dtype=tf.float32)    ,# cv_matrix
                soft_input,  # soft output for this iteration
                scale_factor
            ]
            )
        return soft_output_list
            
    # compute messages from variable nodes to check nodes
    def compute_vc(self,cv_matrix, soft_input,iteration):           
        check_matrix_H = self.code.H
        temp = tf.reduce_sum(cv_matrix,1)                        
        temp = temp+soft_input
        temp = tf.expand_dims(temp,1)
        temp = temp*check_matrix_H
        vc_matrix = temp - cv_matrix
        return vc_matrix 
    # compute messages from check nodes to variable nodes        
    def compute_cv(self,vc_matrix,iteration):
        vc_matrix = tf.clip_by_value(vc_matrix, -10, 10)
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
      
    #combine messages to get posterior LLRs
    def marginalize(self,cv_matrix, soft_input):
        temp = tf.reduce_sum(cv_matrix,1)
        soft_output = temp+soft_input
        return soft_output  
    
    
    def continue_condition(self,soft_input,labels,iteration, cv_matrix,soft_output,scale_factor):
        condition = (iteration < self.layer_iterations) 
        return condition
    
    def belief_propagation_iteration(self,soft_input, labels, iteration, cv_matrix,soft_output,scale_factor):
        # compute vc
        vc_matrix = self.compute_vc(cv_matrix, soft_input,iteration)
        # compute cv
        cv_matrix = self.compute_cv(vc_matrix,iteration)      
        # get output for this iteration
        soft_output = self.marginalize(cv_matrix, soft_input)
        iteration += 1 
        if iteration % scale_factor == 0:
            soft_output_list.append(soft_output)  
              
        return soft_input, labels, iteration, cv_matrix, soft_output,scale_factor