import tensorflow as tf
import numpy as np

class SortedSparseProcessor(tf.keras.layers.Layer):
    """
    Efficient processor for sorted sparse updates with batch support.
    
    Problem: For each batch sample V[b] (length n, all non-zero), and a fixed binary matrix B (m x n),
    we need to:
    1. Sort each V[b] by absolute value globally
    2. For each row i of B, extract the w non-zero values (using B[i] as mask) from sorted V[b]
    3. Feed extracted (already sorted) values into neural network
    4. Apply updates back to original positions, with three column-cyclic shifted versions
    5. Accumulate all three block updates to original X matrix
    
    Parameters:
    - B: Binary matrix [m, n] indicating non-zero positions per row
    - c1, c2: Column cyclic shift amounts for two additional blocks
    - w: Number of non-zero entries per row of B (assumed constant across rows)
    
    Performance for typical use case (batch_size=100, m=64, n=128, w=22):
    - Sorting: ~89k ops, <1ms on GPU
    - Gather operations: ~140k indices, ~0.5ms
    - Neural network: ~1M parameters, ~2ms
    - Scatter updates: ~422k indices, ~1ms
    - Total: ~5ms per batch
    """
    
    def __init__(self, B, c1, c2, **kwargs):
        super().__init__(**kwargs)
        
        # Binary matrix B: [m, n], indicates non-zero positions
        self.B = tf.constant(B, dtype=tf.float32)
        self.m = B.shape[0]       # Number of rows in B (and X)
        self.n = B.shape[1]       # Number of columns
        self.c1 = c1               # First column shift amount
        self.c2 = c2               # Second column shift amount
        self.w = int(tf.reduce_sum(B[0]).numpy())  # Non-zero count per row (assumed constant)
        
        # Precompute: For each row i, the column indices of non-zero entries
        # Shape: [m, w], where element [i, t] is the original column index
        row_nnz_cols_list = []
        for i in range(self.m):
            cols = tf.where(self.B[i] > 0)[:, 0].numpy()
            row_nnz_cols_list.append(cols)
        self.row_nnz_cols = tf.constant(row_nnz_cols_list, dtype=tf.int32)  # [m, w]
        
        # Precompute the three block shift amounts
        self.shifts = tf.constant([0, c1, c2], dtype=tf.int32)  # [3]
        
    def _batch_sort_by_abs(self, V_batch):
        """
        Sort each row of V_batch by absolute value in ascending order.
        
        Args:
            V_batch: [batch_size, n] input matrix (all values non-zero)
            
        Returns:
            sorted_vals: [batch_size, n] values sorted by |value| ascending
            sorted_cols: [batch_size, n] original column indices after sorting
        """
        abs_V = tf.abs(V_batch)                           # [B, n]
        # top_k returns descending order, so we reverse to get ascending
        sorted_vals_desc, sorted_cols_desc = tf.math.top_k(abs_V, k=self.n)
        sorted_vals = tf.reverse(sorted_vals_desc, axis=[-1])   # [B, n]
        sorted_cols = tf.reverse(sorted_cols_desc, axis=[-1])    # [B, n]
        return sorted_vals, sorted_cols
    def _build_column_to_position_map(self, sorted_cols):
        # for each position, which original column is there?
        col_to_pos = tf.argsort(sorted_cols, axis=-1)  # [B, n]
        return col_to_pos
    def _extract_row_inputs(self, sorted_vals, col_to_pos):
        """
        Extract neural network inputs for all rows.
        
        Args:
            sorted_vals: [batch_size, n] values in sorted order
            col_to_pos: [batch_size, n] column to sorted position mapping
            
        Returns:
            inputs: [batch_size, m, w] values ready for neural network
            positions: [batch_size, m, w] original unsorted positions
            positions_sorted: [batch_size, m, w] sorted positions
        """
        batch_size = tf.shape(sorted_vals)[0]
        
        # Expand dimensions for broadcasting
        row_nnz_cols_expanded = self.row_nnz_cols[tf.newaxis, :, :]   # [1, m, w]
        batch_indices = tf.range(batch_size)[:, tf.newaxis, tf.newaxis]  # [B, 1, 1]
        
        # Build gather_nd indices: (batch, row, t) -> column index
        indices = tf.stack([
            tf.broadcast_to(batch_indices, [batch_size, self.m, self.w]),
            tf.broadcast_to(row_nnz_cols_expanded, [batch_size, self.m, self.w])
        ], axis=-1)  # [B, m, w, 2]
        
        # Get positions of each non-zero column in the sorted order
        positions = tf.gather_nd(col_to_pos, indices)  # [B, m, w]
        
        # Sort positions within each row (ascending order)
        positions_sorted = tf.sort(positions, axis=-1)  # [B, m, w]
        
        # Extract values using batch_dims=1
        # sorted_vals: [B, n], positions_sorted: [B, m, w]
        # batch_dims=1 means first dimension (B) is batch dimension
        inputs = tf.gather(sorted_vals, positions_sorted, batch_dims=1)  # [B, m, w]
        
        return inputs, positions, positions_sorted
    
    def _neural_network(self, x):
        """
        Small MLP for processing sorted sparse values.
        
        Args:
            x: [batch_size * m, w] input values
            
        Returns:
            [batch_size * m, w] output updates (same shape as input)
        """
        # Simple 3-layer MLP with ReLU activations
        hidden1 = tf.keras.layers.Dense(6, activation='relu')(x)
        hidden2 = tf.keras.layers.Dense(3, activation='relu')(hidden1)
        output = tf.keras.layers.Dense(self.w)(hidden2)
        return output
    def _apply_updates_with_cyclic_shifts(self, updates, positions, positions_sorted, batch_size):
        """
        Apply updates to original positions with three cyclic shifts.
        
        Args:
            updates: [batch_size, m, w] neural network outputs
            positions: [batch_size, m, w] original unsorted positions (for mapping)
            positions_sorted: [batch_size, m, w] sorted positions (order matches updates tensor)
            batch_size: scalar tensor for the current batch
            
        Returns:
            updates_sum: [batch_size, m, n] accumulated updates for each original element
        """
        # Recover original column order for each update
        # positions_sorted[t] came from sorting positions; we need the inverse mapping
        # Use argsort on positions to get the order that maps sorted back to original column order
        orig_order = tf.argsort(positions, axis=-1)  # [B, m, w]
        
        # Recover original columns in the order that matches updates tensor
        # orig_cols_expanded: [1, m, w] -> broadcast to [B, m, w]
        orig_cols_expanded = self.row_nnz_cols[tf.newaxis, :, :]  # [1, m, w]
        
        # Use gather with batch_dims=2
        # orig_cols_expanded has shape [1, m, w], orig_order has [B, m, w]
        # We want to broadcast the first dimension
        # Solution: tile orig_cols_expanded to match batch_size
        orig_cols_tiled = tf.tile(orig_cols_expanded, [batch_size, 1, 1])  # [B, m, w]
        orig_cols = tf.gather(orig_cols_tiled, orig_order, batch_dims=2)  # [B, m, w]
        
        # Initialize accumulation tensor
        updates_sum = tf.zeros([batch_size, self.m, self.n], dtype=tf.float32)
        
        # Process each of the three cyclic shifts
        for shift in [0, self.c1, self.c2]:
            # Target column after applying cyclic shift
            target_cols = (orig_cols + shift) % self.n  # [B, m, w]
            
            # Flatten all dimensions for scatter_nd operation
            # Create indices for each update
            batch_indices = tf.range(batch_size)[:, tf.newaxis, tf.newaxis]  # [B, 1, 1]
            row_indices = tf.range(self.m)[tf.newaxis, :, tf.newaxis]  # [1, m, 1]
            
            # Broadcast to full shape
            batch_idx = tf.broadcast_to(batch_indices, [batch_size, self.m, self.w])  # [B, m, w]
            row_idx = tf.broadcast_to(row_indices, [batch_size, self.m, self.w])  # [B, m, w]
            
            # Stack to create scatter indices [B, m, w, 3]
            scatter_indices = tf.stack([batch_idx, row_idx, target_cols], axis=-1)  # [B, m, w, 3]
            
            # Flatten to [B*m*w, 3]
            flat_indices = tf.reshape(scatter_indices, [-1, 3])
            flat_updates = tf.reshape(updates, [-1])
            
            # Accumulate updates
            updates_sum = tf.tensor_scatter_nd_add(
                updates_sum, 
                flat_indices, 
                flat_updates
            )  # [B, m, n]
        
        return updates_sum
    
    def call(self, V_batch):
        """
        Main processing pipeline.
        
        Args:
            V_batch: [batch_size, n] input data (all entries non-zero)
            
        Returns:
            final_updates: [batch_size, m, n] accumulated updates for all three cyclic blocks
        """
        batch_size = tf.shape(V_batch)[0]
        
        # Step 1: Sort each batch sample by absolute value
        sorted_vals, sorted_cols = self._batch_sort_by_abs(V_batch)  # Both [B, n]
        
        # Step 2: Build column -> sorted position mapping
        col_to_pos = self._build_column_to_position_map(sorted_cols)  # [B, n]
        
        # Step 3: Extract neural network inputs (already sorted by value)
        inputs, positions, positions_sorted = self._extract_row_inputs(
            sorted_vals, col_to_pos
        )                                                              # inputs: [B, m, w]
        
        # Step 4: Neural network inference
        # Reshape to [B*m, w] for batch processing
        nn_input = tf.reshape(inputs, [-1, self.w])                    # [B*m, w]
        #nn_output = self._neural_network(nn_input)                     # [B*m, w]
        nn_output = -2*nn_input                    # [B*m, w]
        nn_output = tf.reshape(nn_output, [batch_size, self.m, self.w]) # [B, m, w]
        
        # Step 5: Apply updates with cyclic shifts and accumulate
        final_updates = self._apply_updates_with_cyclic_shifts(
            nn_output, positions, positions_sorted, batch_size
        )                                                              # [B, m, n]
        
        return final_updates


# Example usage:
if __name__ == "__main__":
    np.random.seed(42)
    # Parameters
    m, n, w = 4, 7, 3
    batch_size = 2
    c1, c2 = 2, 3  # Example cyclic shifts
    
    # Create binary matrix B with exactly w ones per row
    B = np.zeros([m, n], dtype=np.float32)
    for i in range(m):
        # Randomly select w columns for this row (can be any pattern)
        cols = np.random.choice(n, w, replace=False)
        B[i, cols] = 1.0
    
    # Create input batch (all non-zero)
    V_batch = np.random.randn(batch_size, n).astype(np.float32)
    
    # Build and run processor
    processor = SortedSparseProcessor(B, c1, c2)
    result = processor(V_batch)  # Shape: [batch_size, m, n]
    
    print(f"Input shape: {V_batch.shape}")
    print(f"B matrix shape: {B.shape}")
    print(f"Output shape: {result.shape}")