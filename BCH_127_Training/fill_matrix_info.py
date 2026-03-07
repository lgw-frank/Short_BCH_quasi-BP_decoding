import numpy as np
import globalmap as GL
import galois
import pickle
import random
import os
from scipy.special import comb
from collections import Counter
from collections import defaultdict
#from itertools import cycle, islice
GF2 = galois.GF(2)

def load_txt_matrix(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        matrix = [list(map(int, line.strip(';\n').split())) for line in lines]
    return np.array(matrix, dtype=np.int8)

def load_alist_to_numpy(filepath):
	# parity-check matrix; Tanner graph parameters
    with open(filepath,'rt') as f:
        line= str(f.readline()).strip('\n').split(' ')
		# get n and m (n-k) from first line
        n,m = [int(s) for s in line]      
#################################################################################################################
        var_degrees = np.zeros(n).astype(int) # degree of each variable node    
		# initialize H
        H = np.zeros([m,n]).astype(int)
        line =  str(f.readline()).strip('\n').split(' ')
        line =  str(f.readline()).strip('\n').split(' ')
        line =  str(f.readline()).strip('\n').split(' ')

        var_edges = [[] for _ in range(0,n)]
        for i in range(0,n):
            line =  str(f.readline()).strip('\n').split(' ')
            var_edges[i] = [(int(s)-1) for s in line if s not in ['0','']]
            var_degrees[i] = len(var_edges[i])
            H[var_edges[i], i] = 1
    return H

def load_parity_check_matrix(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.alist':
        return load_alist_to_numpy(filepath)
    elif ext == '.txt':
        return load_txt_matrix(filepath)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
        
class Code:
    def __init__(self,H_filename):
        self.load_code(H_filename) 
        
    def load_code(self,file_path):
        prefix_str = GL.get_map('prefix_str')
         # Load the matrix
        H = load_parity_check_matrix(file_path)
        #print(f'\nH with shape: {H.shape}')  # Check dimensions  
        self.G = self.generator_matrix(H) 
        #effective number of message bits in a codeword        
        self.k = self.G.shape[0] 
        self.n = self.G.shape[1]   
        self.original_H = H
        if GL.get_map('regular_matrix'):
            self.H = H
        else:
           reduction_iteration = GL.get_map('reduction_iteration')
           redundancy_factor = GL.get_map('redundancy_factor')
           absolute_head = f'../{prefix_str.upper()}_'+str(self.n)+'_training/ckpts/'
           saved_file = absolute_head + 'extended_parity_check_matrix_'+str(redundancy_factor)+'_'+str(reduction_iteration)+'.pkl'                       
           os.makedirs(os.path.dirname(saved_file), exist_ok=True)
           unique_H_list = []
           if GL.get_map('generate_extended_parity_check_matrix') and (not os.path.exists(saved_file)):
               # Heuristic optimization using simulated annealing
               # Simulated annealing parameters
               initial_temp = 1000
               cooling_rate = 0.995
               num_iterations = 3000
               # Adjust this factor to balance between minimizing 4-cycles and row weight variation
               beta  = 1 
               #initial reduction of row weight
               target_rank = self.n-self.k
               redundancy_factor = GL.get_map('redundancy_factor') #redundancy multiplier 
               tolerated_max = int(target_rank*redundancy_factor)
               unique_H = self.sparsify_parity_check_matrix(H.copy())
               unique_H_list.append(unique_H)
               for i in range(reduction_iteration):
                   #further reductin of row weigtht
                   unique_H = unique_H[:tolerated_max]
                   unique_H = self.refined_sparsify_parity_check_matrix(unique_H.copy())  
                   unique_H_list.append(unique_H[:tolerated_max])
               #construct full-rank parity-check matrix
               full_rank_matrix = self.add_cyclic_shifts_to_full_rank(unique_H_list,target_rank)        
               target_rows = int(redundancy_factor*(self.n-self.k))
               gap_rows = target_rows-full_rank_matrix.shape[0]
               min_H = full_rank_matrix.copy()
               reordered_matrix = self.sort_rows_by_weight(full_rank_matrix)
               if gap_rows > 0:
                   for i in range(gap_rows):
                       j = i%reordered_matrix.shape[0]
                       min_H = np.vstack([min_H,reordered_matrix[j:j+1]])
               else:
                   min_H = self.reduce_matrix_rank_preserving(reordered_matrix,target_rows) 
                                                          
               H_optimized, _ = self.simulated_annealing_row_operations(min_H, initial_temp, cooling_rate, num_iterations, beta)
               H_optimized = np.unique(H_optimized,axis=0)
               #self.print_matrix_info(H_optimized)
               
               with open(saved_file, 'wb') as file:
                   # Use pickle.dump() to write the object to the file
                   pickle.dump(H_optimized, file) 
           else:    
               with open(saved_file, 'rb') as file:
                   # Use pickle.dump() to write the object to the file
                   #print('Loading PCM from:',saved_file)
                   H_optimized = pickle.load(file) 
                   
           self.H = H_optimized
           print(f"\n Matrix ({H.shape}) before optimization:")
           self.print_matrix_info(H)
           print(f"\n Matrix ({self.H.shape}) after optimization:")
           self.print_matrix_info(self.H)
           
    
    def split_matrix_by_row_weight_sorted(self,matrix):
        """
        Split a binary matrix into submatrices where all rows in each submatrix have the same weight (number of 1s),
        and return them in order of increasing row weight.
        
        Parameters:
            matrix: A 2D list or numpy array representing a binary matrix (containing only 0s and 1s)
            
        Returns:
            A list of submatrices (as numpy arrays) sorted by their row weights in ascending order.
            Each submatrix contains rows that all share the same weight.
        """
        
        # Convert input to numpy array for efficient processing
        matrix = np.array(matrix)
        
        # Calculate row weights (sum of 1s in each row)
        # axis=1 means sum across columns for each row
        row_weights = np.sum(matrix, axis=1)
        
        # Group row indices by their weights using a dictionary
        # Key: weight (int), Value: list of row indices with that weight
        weight_groups = defaultdict(list)
        for idx, weight in enumerate(row_weights):
            weight_groups[weight].append(idx)
        
        # Build the submatrices in order of increasing weight
        submatrices = []
        # Process weights in sorted order to ensure increasing weight sequence
        for weight in sorted(weight_groups.keys()):
            # Get all row indices for the current weight
            indices = weight_groups[weight]
            # Extract the submatrix containing these rows
            submatrix = matrix[indices, :]
            # Add to our result list
            submatrices.append(submatrix)  
        return submatrices

    def detect_rank(self,matrix):
        GF_matrix = GF2(matrix)
        matrix_rank = np.linalg.matrix_rank(GF_matrix)
        return matrix_rank
    
    def print_matrix_info(self,matrix):
       num_cycles = self.count_4_cycles(matrix)
       row_mean, row_std,row_dist = self.row_weight_variation(matrix)
       col_mean,col_std,col_dist = self.column_weight_variation(matrix)
       rank = self.detect_rank(matrix)    
       print(f"Number of 4-cycles for chosen H of shape ({matrix.shape}):{num_cycles} of Rank:{rank}")
       print(f"Row/col weight {row_mean:.1f}/{col_mean:.1f} std:{row_std:.1f}/{col_std:.1f}")
       print(f'Row weight distribution:{row_dist}')
       print(f'Column weight distribution:{col_dist}')
    
            
    def count_4_cycles(self,H):
        rows, cols = H.shape
        num_cycles = 0
   
        # Iterate over pairs of columns
        for i in range(cols):
            for j in range(i + 1, cols):
                common_non_zero_rows = np.where((H[:, i] == 1) & (H[:, j] == 1))[0]
                num_common = len(common_non_zero_rows)
                if num_common >= 2:
                    num_cycles += comb(num_common, 2, exact=True)                   
        return num_cycles

    def weight_variation(self,weights):
        mean_weight = np.mean(weights)
        std = np.std(weights)
        return mean_weight, std
    
    def row_weight_variation(self,H):
        row_weights = np.sum(H, axis=1).tolist()
        row_distribution = Counter(row_weights)
        mean_weight, std = self.weight_variation(row_weights)
        return mean_weight, std,row_distribution
   
    def column_weight_variation(self,H):
        col_weights = np.sum(H, axis=0).tolist()
        col_distribution = Counter(col_weights)
        mean_weight, std = self.weight_variation(col_weights)
        return  mean_weight, std,col_distribution
    
    def cost_function(self,H,beta):
        num_4_cycles = self.count_4_cycles(H)
        _,col_weight_var,_ = self.column_weight_variation(H)       
        return num_4_cycles + beta * col_weight_var
   
    def get_row_cycle_contributions(self,H):
        rows, cols = H.shape
        row_contributions = np.zeros(rows)
        # Iterate over pairs of columns
        for i in range(cols):
            for j in range(i + 1, cols):
                common_non_zero_rows = np.where((H[:, i] == 1) & (H[:, j] == 1))[0]
                num_common = len(common_non_zero_rows)
                if num_common >= 2:
                    for row in common_non_zero_rows:
                        row_contributions[row] += comb(num_common - 1, 1, exact=True)  # Contribution to 4-cycles    
        return row_contributions        
        
    def simulated_annealing_row_operations(self,H, initial_temp, cooling_rate, num_iterations, beta):
       current_temp = initial_temp
       current_H = H.copy()
       current_cost = self.cost_function(current_H, beta)
       best_H = current_H.copy()
       best_cost = current_cost
       target_rank = self.detect_rank(current_H)
   
       for i in range(num_iterations):
           print('.',end='')
           if (i+1)%100 == 0:
               print(f'\n{i+1}th\n')
           if current_temp <= 0:
               break
           # Create a new candidate solution
           new_H = current_H.copy()
           #row shifts for cyclic codes
           # Calculate row contributions to 4-cycles
           row_contributions = self.get_row_cycle_contributions(new_H)
           # Select rows based on their contributions
           if np.sum(row_contributions)==0:
               best_H = current_H
               best_cost = self.cost_function(current_H,beta)
               break
           row_probs = row_contributions / np.sum(row_contributions)
           row = np.random.choice(new_H.shape[0], p=row_probs)
           shift_amount = np.random.randint(0, new_H.shape[1])
           new_H[row] = np.roll(new_H[row], shift_amount)
           new_cost = self.cost_function(new_H,beta)        
           # Accept new solution if it's better or with a certain probability if it's worse
           if (new_cost < best_cost  or random.uniform(0, 1) < np.exp((current_cost - new_cost) / current_temp)) and self.detect_rank(new_H)==target_rank:
               current_H = new_H.copy()
               current_cost = new_cost          
           if current_cost < best_cost:
               print(current_cost)
               best_H = current_H.copy()
               best_cost = current_cost 
           # Cool down
           current_temp *= cooling_rate  
       return best_H, best_cost
        
    def sort_rows_by_weight(self, matrix):
        """
        Sort rows of a binary matrix in ascending order of their Hamming weight.
        
        Parameters:
            matrix (numpy.ndarray): Binary matrix (2D array) containing only 0s and 1s.
            
        Returns:
            numpy.ndarray: Matrix with rows sorted by Hamming weight in ascending order.
            
        Note:
            Hamming weight = number of 1s in a row
        """
        # Calculate Hamming weights of each row (sum of 1s along axis=1)
        row_weights = np.sum(matrix, axis=1)
        
        # Get indices that would sort the weights in ascending order
        # (default behavior of argsort is ascending)
        sorted_indices = np.argsort(row_weights)
        
        # Reorder rows based on sorted indices
        sorted_matrix = matrix[sorted_indices]
        
        return sorted_matrix

    
    def reduce_matrix_rank_preserving(self,matrix,target_rows):
        """
        Reduce a binary matrix by removing rows from the bottom while preserving the original rank.
        
        Algorithm:
        1. Compute the original rank of the matrix
        2. Iterate from the last row to the first:
           a. Remove the current row temporarily
           b. If the reduced matrix maintains the original rank, keep the deletion
           c. Otherwise, restore the row
        3. Return the maximally reduced matrix
        
        Parameters:
            matrix (numpy.ndarray): A binary matrix (elements in {0,1})
            
        Returns:
            numpy.ndarray: The reduced matrix with same rank as original
        """
        # Convert to numpy array if not already
        matrix = np.array(matrix, dtype=int)
        
        # Handle empty matrix case
        if matrix.size == 0:
            return matrix
        
        # Compute original rank using Gaussian elimination
        original_rank = self.detect_rank(matrix) 
        reduced_matrix = matrix.copy()
        
        # Iterate from last row to first
        for i in range(len(reduced_matrix)-1, -1, -1):
            # Early termination condition
            if len(reduced_matrix) == original_rank or len(reduced_matrix) == target_rows:
                break
            # Temporarily remove the i-th row
            temp_matrix = np.delete(reduced_matrix, i, axis=0)
          
            # Check if reduced matrix maintains rank
            if temp_matrix.size > 0 and self.detect_rank(temp_matrix) == original_rank:
                reduced_matrix = temp_matrix
            # Else: keep the row in the matrix
        
        return reduced_matrix


    def get_canonical_rotation(self,arr):
        n = len(arr)
        doubled = np.tile(arr, 2)
        # Convert rotations to strings for comparison
        rotations = [''.join(map(str, doubled[i:i+n])) for i in range(n)]
        min_idx = np.argmin(rotations)
        return tuple(doubled[min_idx:min_idx+n])
    
    def remove_circular_duplicates(self,matrix):
        np_matrix = np.unique(matrix, axis=0)
        canonical_forms = set()
        unique_rows = []     
        for row in np_matrix:
            canonical = tuple(self.get_canonical_rotation(row))
            if canonical not in canonical_forms:
                canonical_forms.add(canonical)
                unique_rows.append(row)      
        return np.reshape(unique_rows,[-1,matrix.shape[1]])
                              
    def lexicographically_smallest_rotation(self,row):
        """Find the lexicographically smallest rotation of a row."""
        rotations = [np.roll(row, i) for i in range(len(row))]
        return min(rotations, key=lambda x: tuple(x))
    
    def remove_circular_duplicates2(self,tensor):
        """Remove duplicate rows considering circular shifts."""
        tensor = np.unique(tensor, axis=0)
        normalized_rows = np.array([self.lexicographically_smallest_rotation(row) for row in tensor])
        unique_matrix = np.unique(normalized_rows, axis=0)
        #counter_pro = Counter(np.sum(unique_matrix,1))
        #print('H up to shifts with shape:',sorted(counter_pro.items()))
        return unique_matrix
    
    def iterate_for_min_weight_rows(self,matrix_cmp,range_start):
        base_row = matrix_cmp[range_start:range_start+1]
        new_row_list = []
        minimal_value = np.sum(base_row)
        for shift in range(matrix_cmp.shape[1]):          
            matrix_sum = (base_row + np.roll(matrix_cmp,shift,axis=1))%2
            row_sum = np.sum(matrix_sum,1)
            # Check if the minimum value is zero
            # Find the minimum value of the list
            min_non_zero = np.min(row_sum[row_sum!=0])
            if min_non_zero <= np.sum(base_row):   
                minimal_value = min_non_zero
                # Get the indices of the minimum non-zero elements
                indices = np.where(row_sum == min_non_zero)[0]
                for j in indices:
                    element_tuple = (minimal_value,matrix_sum[j])
                    new_row_list.append(element_tuple)
        if minimal_value == np.sum(base_row):
            new_row_list.append((minimal_value,base_row.flatten()))
                    
        #check and keep the min elements only
        unique_filtered_list = []
        if new_row_list:
            # Find the minimum value with respect to the first element of all tuples
            min_value = min(new_row_list, key=lambda x: x[0])[0]          
            # Filter the list to include only tuples that have this minimum value
            filtered_list = [tup[1] for tup in new_row_list if tup[0] == min_value]
            seen = set()   
            for element in filtered_list:
                if tuple(element) not in seen:
                    unique_filtered_list.append(element)
                    seen.add(tuple(element))
        return unique_filtered_list

    def generate_all_cyclic_shifts(self,matrix):
        shift_limit = matrix.shape[1]
        shifted_row_list = []
        sub_matrices = self.split_matrix_by_row_weight_sorted(matrix)
        for sub_matrix in sub_matrices:
            for shift in range(shift_limit):
                for row in sub_matrix:          
                    shifted_row_list.append(np.roll(row, shift))
        dilated_matrix = np.vstack(shifted_row_list)
        _, unique_indices = np.unique(dilated_matrix, axis=0, return_index=True)
        preserved_order_unique = dilated_matrix[np.sort(unique_indices)]
        return preserved_order_unique


    def add_cyclic_shifts_to_full_rank(self,matrix_list,target_rank):
        macro_matrix = np.concatenate(matrix_list,axis=0)
        unique_macro_matrix = np.unique(macro_matrix,axis=0)
        row_groups,weight_list = self.group_rows_by_weight_acquire_rank(unique_macro_matrix)
        base_matrix = row_groups[weight_list[0]]
        current_rank = self.detect_rank(base_matrix)
        if current_rank < target_rank:
            early_terminator = False
            for weight in weight_list:  
                element_matrix = row_groups[weight]
                dilated_matrix = self.generate_all_cyclic_shifts(element_matrix)
                for new_row in dilated_matrix:
                    if any(np.array_equal(new_row, row) for row in base_matrix):
                        continue
                    tmp_matrix = np.vstack([base_matrix, new_row])
                    new_rank = self.detect_rank(tmp_matrix)           
                    if new_rank > current_rank:
                        current_rank = new_rank
                        base_matrix = tmp_matrix
                        #print(f"Added row: {new_row}, new rank: {new_rank}")
                        if new_rank == target_rank:
                            early_terminator = True
                            break
                if early_terminator:
                    break
        return base_matrix
    
    def row_reduce_gf(self,matrix):
        GF_matrix = GF2(matrix)
        Ref_matrix = GF_matrix.row_reduce()
        Ref_rank = np.linalg.matrix_rank(Ref_matrix)
        return Ref_matrix,Ref_rank
    
    def sparsify_parity_check_matrix(self,H):
        Ref_matrix,Ref_rank = self.row_reduce_gf(H)
        #recover original reduced echelon form
        original_ref_H = Ref_matrix.view(np.ndarray)
        #print('Initial summary:')
        #print(f'Original rank:{Ref_rank}')
        new_row_list = []
        for i in range(original_ref_H.shape[0]):
            tmp_matrix = (Ref_matrix[i:i+1] + Ref_matrix).view(np.ndarray)
            # Find the minimum of non-zero elements
            row_sum = np.sum(tmp_matrix,1)
            min_non_zero = np.min(row_sum[row_sum!=0])
            if min_non_zero <= np.sum(original_ref_H[i]):
                # Get the indices of the minimum non-zero elements
                indices = np.where(row_sum == min_non_zero)[0]
                for j in indices:
                    new_row_list.append(tmp_matrix[j])
            if min_non_zero >= np.sum(original_ref_H[i]):
                new_row_list.append(original_ref_H[i])
        reduce_H_matrix = np.reshape(new_row_list,[-1,original_ref_H.shape[1]])
        # remove duplicated and shifted rows
        unique_H_matrix = self.remove_circular_duplicates(reduce_H_matrix)
        unique_H_matrix = self.sort_rows_by_weight(unique_H_matrix)
        #verify ths syndrome equal to all-zero matrix
        syndrome_result = unique_H_matrix.dot(self.G.T)%2
        if np.all(syndrome_result==0):
          print("Initial parity checks passed !")
        else:
          print("Something wrong happened, parity check failed !")  
        return unique_H_matrix
    
    def group_rows_by_weight_acquire_rank(self,matrix):
        """
        Group rows of a binary matrix based on their Hamming weight.   
        Parameters:
        matrix (numpy.ndarray): Binary matrix (2D array).
        Returns:
        dict: Keys are weights, and values are submatrices with rows of the same weight.
        """
        ordered_matrix = self.sort_rows_by_weight(matrix)
        # Calculate Hamming weight for each row
        row_weights = np.sum(ordered_matrix, axis=1)       
        # Group rows by their weight
        row_groups = {}
        weight_list = []
        for weight in np.sort(np.unique(row_weights)):
            weight_list.append(weight)
            # Find indices of rows with the current weight
            indices = np.where(row_weights == weight)[0]
            # Extract corresponding rows
            row_groups[weight] = ordered_matrix[indices]    
        # Print results
        # print('summary:Rank dist of varied Hamming Weights:',end=' ')
        # for weight, submatrix in weight_groups.items():
        #     rank_value = np.linalg.matrix_rank(GF2(submatrix))
        #     print(f"(W:{weight},R:{rank_value})",end=' ')
        # rank_value = np.linalg.matrix_rank(GF2(matrix))
        # print(f'\nTotal Rank:{rank_value}')
        return row_groups,weight_list
    
    
    def refined_sparsify_parity_check_matrix(self,unique_ref_H):
        new_matrix_list = []
        for i in range(unique_ref_H.shape[0]):
            range_start = i
            #print(i,end=' ')
            row_candidates = self.iterate_for_min_weight_rows(unique_ref_H,range_start)
            new_matrix_list = new_matrix_list + row_candidates
        reduce_H_matrix = np.reshape(new_matrix_list,[-1,self.n])      
        # remove duplicated and shifted rows
        unique_ref_H = self.remove_circular_duplicates(reduce_H_matrix)
        syndrome_result = unique_ref_H.dot(self.G.T)%2
        if np.all(syndrome_result==0):
          print("Advanced parity checks passed !")
        else:
          print("Something wrong happened, parity check failed !")  
        #self.group_rows_by_weight_acquire_rank(unique_ref_H)
        unique_ref_H = self.sort_rows_by_weight(unique_ref_H)
        print('')
        return unique_ref_H   

    def generator_matrix(self,parity_check_matrix):
          # H assumed to be full row rank to obtain its systematic form
          tmp_H = np.copy(parity_check_matrix)
          #reducing into row-echelon form and record column 
          #indices involved in swapping
          row_echelon_form,record_col_exchange_index = self.gf2elim(tmp_H)
          H_shape = row_echelon_form.shape
          # H is reduced into [I H_2]
          split_H = np.hsplit(row_echelon_form,(H_shape[0],H_shape[1])) 
          #Generator matrix in systematic form [H_2^T I] in GF(2)
          G1 = split_H[1].T
          G2 = np.identity(H_shape[1]-H_shape[0],dtype=int)
          G = np.concatenate((G1,G2),axis=1)
          #undo the swapping of columns in reversed order
          for i in reversed(range(len(record_col_exchange_index))):
              temp = np.copy(G[:,record_col_exchange_index[i][0]])
              G[:,record_col_exchange_index[i][0]] = \
                  G[:,record_col_exchange_index[i][1]]
              G[:,record_col_exchange_index[i][1]] = temp
          #verify ths syndrome equal to all-zero matrix
          #Syndrome_result = parity_check_matrix.dot(G.T)%2
          # if np.all(Syndrome_result==0):
          #   print(f'Generator matrix created successfully with shape:{G.shape}')
          # else:
          #   print("Something wrong happened, generator matrix failed to be valid")     
          return G  
      
    def gf2elim(self,M):
          m,n = M.shape
          i=0
          j=0
          record_col_exchange_index = []
          while i < m and j < n:
              #print(M)
              # find value and index of largest element in remainder of column j
              if np.max(M[i:, j]):
                  k = np.argmax(M[i:, j]) +i
            # swap rows
                  #M[[k, i]] = M[[i, k]] this doesn't work with numba
                  if k !=i:
                      temp = np.copy(M[k])
                      M[k] = M[i]
                      M[i] = temp              
              else:
                  if not np.max(M[i, j:]):
                      M = np.delete(M,i,axis=0) #delete a all-zero row which is redundant
                      m = m-1  #update according info
                      continue
                  else:
                      column_k = np.argmax(M[i, j:]) +j
                      temp = np.copy(M[:,column_k])
                      M[:,column_k] = M[:,j]
                      M[:,j] = temp
                      record_col_exchange_index.append((j,column_k))
          
              aijn = M[i, j:]
              col = np.copy(M[:, j]) #make a copy otherwise M will be directly affected
              col[i] = 0 #avoid xoring pivot row with itself
              flip = np.outer(col, aijn)
              M[:, j:] = M[:, j:] ^ flip
              i += 1
              j +=1
          return M,record_col_exchange_index          

