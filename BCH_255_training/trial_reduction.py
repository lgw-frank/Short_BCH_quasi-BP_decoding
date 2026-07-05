# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 20:02:04 2024

@author: zidonghua_30
"""

import numpy as np

def load_alist(file_path):
    """Load a parity check matrix from an .alist file."""
    with open(file_path, 'r') as f:
        line= str(f.readline()).strip('\n').split(' ')
		# get n and m (n-k) from first line
        n,m = [int(s) for s in line]
        #assigned manually for redundant check matrix otherwise
       
#################################################################################################################
        var_degrees = np.zeros(n).astype(int) # degree of each variable node
		# initialize H
        H = np.zeros([m,n]).astype(int)
        line =  str(f.readline()).strip('\n').split(' ')
        max_var_degree, max_chk_degree = [int(s) for s in line]
        line =  str(f.readline()).strip('\n').split(' ')
        line =  str(f.readline()).strip('\n').split(' ')

        var_edges = [[] for _ in range(0,n)]
        for i in range(0,n):
            line =  str(f.readline()).strip('\n').split(' ')
            var_edges[i] = [(int(s)-1) for s in line if s not in ['0','']]
            var_degrees[i] = len(var_edges[i])
            H[var_edges[i], i] = 1

    return H

def row_reduce_gf2(matrix):
    """Perform row reduction on a matrix over GF(2)."""
    m, n = matrix.shape
    matrix = matrix.copy()
    pivot_row = 0
    col_permutations = np.arange(n)

    for pivot_col in range(n):
        if pivot_row >= m:
            break
        max_row = pivot_row + np.argmax(matrix[pivot_row:, pivot_col])
        if matrix[max_row, pivot_col] == 0:
            continue

        # Swap rows
        matrix[[pivot_row, max_row]] = matrix[[max_row, pivot_row]]

        for r in range(pivot_row + 1, m):
            if matrix[r, pivot_col] == 1:
                matrix[r] = (matrix[r] + matrix[pivot_row]) % 2

        pivot_row += 1

    return matrix, col_permutations

def greedy_sparsify(matrix):
    """Greedily sparsify a matrix over GF(2) while preserving its null space."""
    m, n = matrix.shape
    matrix = matrix.copy()
    
    # Ensure matrix is in row echelon form
    matrix, col_permutations = row_reduce_gf2(matrix)
    
    # Iteratively attempt to zero out elements while preserving row echelon form
    for i in range(m):
        for j in range(n):
            if matrix[i, j] == 1:
                temp_matrix = matrix.copy()
                temp_matrix[i, j] = 0
                
                # Check if setting this element to 0 preserves the null space
                temp_matrix_reduced, _ = row_reduce_gf2(temp_matrix)
                matrix_reduced, _ = row_reduce_gf2(matrix)
                if np.array_equal(temp_matrix_reduced, matrix_reduced):
                    matrix = temp_matrix

    return matrix, col_permutations

def main():
    file_path = './BCH_63_36_5_strip.alist'
    H_original = load_alist(file_path)

    print("Original Parity Check Matrix:\n", H_original)
    
    H_sparsified, col_permutations = greedy_sparsify(H_original)

    print("Sparsified Parity Check Matrix:\n", H_sparsified)
    print("Column Permutations:\n", col_permutations)

if __name__ == "__main__":
    main()
