import numpy as np

def load_code(H_filename):
    with open(H_filename, 'rt') as f:
        line = str(f.readline()).strip('\n').split(' ')
        n, m = [int(s) for s in line]
        
        var_degrees = np.zeros(n).astype(int)
        chk_degrees = np.zeros(m).astype(int)
        
        H = np.zeros([m, n]).astype(int)
        line = str(f.readline()).strip('\n').split(' ')
        max_var_degree, max_chk_degree = [int(s) for s in line]
        line = str(f.readline()).strip('\n').split(' ')
        var_degree_dist = [int(s) for s in line[0:-1]] 
        line = str(f.readline()).strip('\n').split(' ')
        chk_degree_dist = [int(s) for s in line[0:-1]]
        
        var_edges = [[] for _ in range(n)]
        for i in range(n):
            line = str(f.readline()).strip('\n').split(' ')
            var_edges[i] = [(int(s)-1) for s in line if s not in ['0', '']]
            var_degrees[i] = len(var_edges[i])
            for j in var_edges[i]:
                H[j, i] = 1

    return H

def find_cycles(H):
    cycles = []
    m, n = H.shape

    for i in range(m):
        for j in range(i+1, m):
            common_cols = np.where(H[i] & H[j])[0]
            if len(common_cols) > 1:
                for idx1 in range(len(common_cols)):
                    for idx2 in range(idx1 + 1, len(common_cols)):
                        cycle = [(i, common_cols[idx1]), (j, common_cols[idx1]), (i, common_cols[idx2]), (j, common_cols[idx2])]
                        cycles.append(cycle)
    return cycles

def break_cycles(H, cycles):
    for cycle in cycles:
        for (row, col) in cycle:
            if H[row, col] == 1:
                # Attempt to break the cycle with row operations
                for k in range(H.shape[0]):
                    if k != row and H[k, col] == 1:
                        H[row] = (H[row] + H[k]) % 2
                        break
                else:
                    # If row operations are not sufficient, use column operations
                    for l in range(H.shape[1]):
                        if l != col and H[row, l] == 1:
                            H[:, col] = (H[:, col] + H[:, l]) % 2
                            break
    return H

def reduce_to_sparse(H):
    rows, cols = H.shape
    for i in range(min(rows, cols)):
        if H[i, i] == 0:
            for j in range(i + 1, rows):
                if H[j, i] == 1:
                    H[[i, j]] = H[[j, i]]  # Swap rows
                    break
        for j in range(i + 1, rows):
            if H[j, i] == 1:
                H[j] = (H[j] + H[i]) % 2  # Add row i to row j

    return H

def sparsify_parity_check_matrix(H_filename):
    H = load_code(H_filename)
    print("Initial H matrix:\n", H)
    
    cycles = find_cycles(H)
    print(f"Number of cycles found: {len(cycles)}")
    if cycles:
        H = break_cycles(H, cycles)
        print("H matrix after breaking cycles:\n", H)
    
    H = reduce_to_sparse(H)
    print("Sparse H matrix:\n", H)
    
    return H

# Usage
H_filename = 'BCH_63_36_5_strip.alist'
H_sparse = sparsify_parity_check_matrix(H_filename)
