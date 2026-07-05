import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import globalmap as GL
from scipy.stats import gaussian_kde
#import matplotlib.cm as cm
import os,pickle
import glob
from pathlib import Path
#@tf.function
def mutual_information_from_llr(llr, bits,weight=1.):
    code = GL.get_map('code_parameters')
    code_parity_check_H = tf.cast(code.H,tf.float32)
    """
    Compute I(L;X) from LLR samples and transmitted bits.
    No all-zero assumption.

    llr  : Tensor of LLR samples
    bits : Corresponding transmitted bits (0/1)
    """
    llr = tf.cast(llr, tf.float32)
    bits = tf.cast(bits, tf.float32)
    expand_bits = tf.expand_dims(bits,axis=1)
    # Sign correction: (-1)^X * L
    llr_corr = (1.0 - 2.0 * expand_bits) * llr
    mi_matrix = (1.-tf.math.softplus(-llr_corr)/ tf.math.log(2.0))*code_parity_check_H
    non_zeros_elements = tf.math.count_nonzero(code.H)
    number_samples = mi_matrix.shape[0]
    #collapsed realizations
    mi_cr = tf.reduce_mean(mi_matrix,axis=0) 
    #collapsed edges
    mi_ce = tf.reduce_sum(mi_matrix,axis=[1,2])/tf.cast(non_zeros_elements, tf.float32) 
    #collapse both(edges+realizations)
    mi_cre = tf.reduce_sum(mi_cr)/tf.cast(non_zeros_elements, tf.float32)
    mi_cer= tf.reduce_sum(mi_ce) /tf.cast(number_samples, tf.float32)     
    return mi_cr,mi_ce,mi_cre,mi_cer

class ScatterEXIT:
    """
    Decoder-agnostic Scatter EXIT (S-EXIT) collector
    """

    def __init__(self):
        self.IA = []
        self.IE = []

    def clear(self):
        self.IA.clear()
        self.IE.clear()
        
    def record_vn(self, llr_extrinsic, bits):
        """
        Record VN extrinsic MI samples
        """
        mi_cr,mi_ce,mi_cre,mi_cer = mutual_information_from_llr(llr_extrinsic, bits)
        self.IA.append((mi_cr,mi_ce,mi_cre,mi_cer))

    def record_cn(self, llr_extrinsic, bits):
        """
        Record CN extrinsic MI samples
        """
        mi_cr,mi_ce,mi_cre,mi_cer = mutual_information_from_llr(llr_extrinsic, bits)
        self.IE.append((mi_cr,mi_ce,mi_cre,mi_cer))
    def proportion_below_threshold(self,tensor, threshold, return_count=False):
        """
        Calculate the proportion of elements in a tensor that are below a given threshold.
        
        Args:
            tensor (tf.Tensor or np.ndarray): Input tensor or array
            threshold (float): Threshold value for comparison
            return_count (bool): If True, also return count of elements below threshold
        
        Returns:
            float: Proportion of elements below threshold (0.0 to 1.0)
            or tuple: (proportion, count) if return_count=True
        
        Examples:
            >>> tensor = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
            >>> proportion_below_threshold(tensor, 3.5)
            0.6  # 3 out of 5 elements are below 3.5
        """
        # Ensure tensor is a TensorFlow tensor
        if not isinstance(tensor, tf.Tensor):
            tensor = tf.convert_to_tensor(tensor)
        
        # Ensure threshold is numeric
        threshold = tf.cast(threshold, tensor.dtype)
        
        # Method 1: Direct calculation
        below_mask = tf.less(tensor, threshold)  # tensor < threshold
        count_below = tf.reduce_sum(tf.cast(below_mask, tf.float32))
        total_elements = tf.cast(tf.size(tensor), tf.float32)
        proportion = count_below / total_elements
        tf.print(count_below)
        if return_count:
            return proportion.numpy(), int(count_below.numpy())
        return round(float(proportion.numpy()),4)  
    
    def get_nonzero_pairs_stack(self,tensor1, tensor2):
            mask = (tensor1 != 0) & (tensor2 != 0)
            indices = tf.where(mask)
        
            values1 = tf.gather_nd(tensor1, indices)
            values2 = tf.gather_nd(tensor2, indices)
            
            indices_f = tf.cast(indices, tf.float32)
            values1_f = tf.cast(values1, tf.float32)
            values2_f = tf.cast(values2, tf.float32)
            result = tf.stack([
                indices_f[:, 0],
                indices_f[:, 1], 
                values1_f,
                values2_f
            ], axis=1)   
            return result

    
    def plot_density_strips_regular_file(self, data_dir_pattern, save_path=None, dpi=800, 
                                       figsize=(20, 8), bandwidth=0.05, max_files=5):
        
        # Find all data files
        if os.path.isdir(data_dir_pattern):
            search_pattern = os.path.join(data_dir_pattern, "*.pkl")
        else:
            search_pattern = data_dir_pattern
        
        data_file = sorted(glob.glob(search_pattern))[:max_files]
        
        if not data_file:
            raise ValueError(f"No data files found: {search_pattern}")
        if len(data_file) > 1:
            raise ValueError(f"More than data files found: {search_pattern}")
        
        try:
            with open(data_file[0], 'rb') as f:
                final_ia_list,final_ie_list = pickle.load(f)
        except Exception as e:
            print(f"Error loading {data_file}: {e}") 
        num_iterations = GL.get_map('num_iterations')
        threshold_MI = GL.get_map('threshold_MI')
        IA_collapsed_edge_realization = [round(float(tf.reduce_mean(final_ia_list[i][2],axis=0)),4)for i in range(len(final_ia_list))]
        IE_collapased_edge_realization = [round(float(tf.reduce_mean(final_ie_list[i][2],axis=0)),4)for i in range(len(final_ie_list))]
        FER_estimate = [self.proportion_below_threshold(tf.reshape(final_ia_list[i][1],[-1]),threshold_MI)   for i in range(num_iterations)]
        tf.print(f'IA:{IA_collapsed_edge_realization}')
        tf.print(f'IE:{IE_collapased_edge_realization}')    
        tf.print(f'FER Estimate:{FER_estimate}') 
        # Extract data
        IA_cr= [tf.reduce_mean(final_ia_list[i][0],axis=0) for i in range(num_iterations)]
        IE_cr = [tf.reduce_mean(final_ie_list[i][0],axis=0) for i in range(num_iterations)]
        IA_ce= [tf.reduce_mean(final_ia_list[i][1],axis=0) for i in range(num_iterations)]
        IE_ce = [tf.reduce_mean(final_ie_list[i][1],axis=0)for i in range(num_iterations)]
        IA_cre = [round(float(tf.reduce_mean(final_ia_list[i][2],axis=0)),4) for i in range(num_iterations)]
        IE_cre = [round(float(tf.reduce_mean(final_ie_list[i][2],axis=0)),4)  for i in range(num_iterations)]
        IA_cer = [round(float(tf.reduce_mean(final_ia_list[i][3],axis=0)),4) for i in range(num_iterations)]
        IE_cer = [round(float(tf.reduce_mean(final_ie_list[i][3],axis=0)),4)  for i in range(num_iterations)]        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        epsilon = 1e-10
        # Left panel: Probability density strips
        for i in range(num_iterations):
            combined_pair_element = self.get_nonzero_pairs_stack(
                IA_cr[i], 
                IE_cr[i]
            )
            
            # Need sufficient data points for meaningful KDE
            if combined_pair_element.shape[0] > 10:
                IA_list = combined_pair_element[:, -2]
                IE_list = combined_pair_element[:, -1]
                
                # Kernel Density Estimation for IA
                if len(IA_list) > 1:
                    kde_ia = gaussian_kde(IA_list, bw_method=bandwidth)
                    x_ia = np.linspace(0, 1, 100)  # MI value range
                    y_ia = kde_ia(x_ia)  # Density values
                    y_ia = y_ia / (y_ia.max() + epsilon) * 0.4  # Normalize to width 0.4
                    
                    # Fill IA density strip (left side of iteration)
                    ax1.fill_betweenx(x_ia, i - y_ia, i, 
                                     color='magenta', alpha=0.5, edgecolor='darkmagenta',
                                     label='IA Distribution' if i == 0 else "")
                
                # Kernel Density Estimation for IE
                

                if len(IE_list) > 1:
                    kde_ie = gaussian_kde(IE_list, bw_method=bandwidth)
                    x_ie = np.linspace(0, 1, 100)  # MI value range
                    y_ie = kde_ie(x_ie)  # Density values
                    y_ie = y_ie / (y_ie.max() + epsilon) * 0.4 # Normalize to width 0.4
                    
                    # Fill IE density strip (right side of iteration)
                    ax1.fill_betweenx(x_ie, i, i + y_ie,
                                     color='blue', alpha=0.5, edgecolor='darkblue',
                                     label='IE Distribution' if i == 0 else "")
        
        # Plot average trajectories
        iterations = range(num_iterations)
        ax1.plot(iterations, IA_cre,
                 'r-', linewidth=1.5, marker='o', markersize=8, label='IA Average')
        ax1.plot(iterations, IE_cre,
                 'c-', linewidth=1.5, marker='s', markersize=8, label='IE Average')
        
        ax1.set_xlabel("Iteration Index", fontsize=12)
        ax1.set_ylabel("Mutual Information (MI)", fontsize=12)
        ax1.grid(True, which='major', alpha=0.4, linestyle='--', linewidth=0.8)
        ax1.grid(True, which='minor', alpha=0.3, linestyle=':', linewidth=0.4)
        ax1.minorticks_on()
        ax1.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
        ax1.set_xlim(-0.5, num_iterations - 0.5)
        ax1.set_ylim(0, 1)
        ax1.legend(loc=(0.7,0.3), fontsize=12)
# Right panel: collapsed edges visualization
# Prepare all data points for combined density visualization 
        for i in range(num_iterations):
                IA_list = IA_ce[i]
                IE_list = IE_ce[i]               
                # Kernel Density Estimation for IA
                if len(IA_list) > 1:
                    kde_ia = gaussian_kde(IA_list, bw_method=bandwidth)
                    x_ia = np.linspace(0, 1, 100)  # MI value range
                    y_ia = kde_ia(x_ia)  # Density values
                    y_ia = y_ia / (y_ia.max() + epsilon) * 0.4  # Normalize to width 0.4
                    
                    # Fill IA density strip (left side of iteration)
                    ax2.fill_betweenx(x_ia, i - y_ia, i, 
                                     color='magenta', alpha=0.5, edgecolor='darkmagenta',
                                     label='IA Distribution' if i == 0 else "")
                
                # Kernel Density Estimation for IE
                if len(IE_list) > 1:
                    kde_ie = gaussian_kde(IE_list, bw_method=bandwidth)
                    x_ie = np.linspace(0, 1, 100)  # MI value range
                    y_ie = kde_ie(x_ie)  # Density values
                    y_ie = y_ie / (y_ie.max() + epsilon)* 0.4  # Normalize to width 0.4
                    
                    # Fill IE density strip (right side of iteration)
                    ax2.fill_betweenx(x_ie, i, i + y_ie,
                                     color='blue', alpha=0.5, edgecolor='darkblue',
                                     label='IE Distribution' if i == 0 else "")
        # Plot average trajectories
        ax2.plot(iterations, IA_cer,
                 'r-', linewidth=1.5, marker='o', markersize=8, label='IA Average')
        ax2.plot(iterations, IE_cer,
                 'c-', linewidth=1.5, marker='s', markersize=8, label='IE Average')
        
        ax2.set_xlabel("Iteration Index", fontsize=12)
        ax2.set_ylabel("Mutual Information (MI)", fontsize=12)
        ax2.grid(True, which='major', alpha=0.4, linestyle='--', linewidth=0.8)
        ax2.grid(True, which='minor', alpha=0.3, linestyle=':', linewidth=0.4)
        ax2.minorticks_on()
        ax2.yaxis.set_minor_locator(plt.MultipleLocator(0.1))  # Y轴每0.1一个minor tick
        ax2.set_xlim(-0.5, num_iterations - 0.5)
        ax2.set_ylim(0, 1)
        ax2.legend(loc=(0.7,0.3),fontsize=12)        
        plt.tight_layout()
        
        # Save or display
        path = Path(data_file[0])
        filename = path.stem
        if save_path:
            self._save_figure(fig, save_path+filename, dpi, 'png')
        plt.show()

    def _save_figure(self, fig, save_path, dpi=300, file_format='png'):
        """
        Internal helper function to save matplotlib figures.
        
        Args:
            fig: matplotlib figure object
            save_path: Path to save the figure
            dpi: Resolution for saved figure
            file_format: Format for saved figure
        """
        import os
        
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Ensure file extension matches format
        if not save_path.lower().endswith(f'.{file_format.lower()}'):
            save_path = f"{os.path.splitext(save_path)[0]}.{file_format}"
        
        # Save figure
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        print(f"Figure saved to: {save_path} (dpi={dpi}, format={file_format})")

    def plot_density_strips_comparison(self, data_dir_pattern, save_path=None, dpi=300, 
                                    figsize=(14, 6), bandwidth=0.1):
        """
        Plot density strips comparison from multiple data files.
        
        This method finds all data files matching the pattern, loads them,
        and plots comparative density strips for IA and IE values at each iteration.
        
        Args:
            data_dir_pattern (str): Pattern or directory path to find data files.
                                Can include wildcards (e.g., 'data/*.pkl' or 'results/experiment_*').
            save_path (str, optional): Path to save the figure. If None, figure is displayed.
            dpi (int): Resolution for saved figure (dots per inch). Default: 300.
            figsize (tuple): Figure size as (width, height) in inches. Default: (14, 6).
            bandwidth (float): Bandwidth parameter for kernel density estimation.
        
        Returns:
            tuple: (matplotlib.figure.Figure, dict) The figure and loaded data dictionary.
        """

        
        # Find all data files matching the pattern
        if os.path.isdir(data_dir_pattern):
            # If a directory is provided, look for all pickle files in it
            search_pattern = os.path.join(data_dir_pattern, "*.pkl")
        else:
            # Use the pattern as-is (could include wildcards)
            search_pattern = data_dir_pattern
        
        data_files = sorted(glob.glob(search_pattern))
        
        if not data_files:
            raise ValueError(f"No data files found matching pattern: {search_pattern}")
        
        print(f"Found {len(data_files)} data files:")
        for i, file_path in enumerate(data_files):
            print(f"  {i+1}. {os.path.basename(file_path)}")
        
        # Load data from all files
        loaded_data = {}
        for file_path in data_files:
            file_name = os.path.basename(file_path)
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    loaded_data[file_name] = data
                print(f"✓ Loaded: {file_name}")
            except Exception as e:
                print(f"✗ Failed to load {file_name}: {e}")
        
        if not loaded_data:
            raise ValueError("No data files could be loaded successfully")
        
        # Get common parameters
        num_iterations = GL.get_map('num_iterations')
        
        # Define color palette for different files
        colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(loaded_data))))
        line_styles = ['-', '--', '-.', ':']
        marker_styles = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Left panel: CR/RE edges
        for idx, (file_name, combinde_list) in enumerate(loaded_data.items()):
            color = colors[idx % len(colors)]
            line_style = line_styles[idx % len(line_styles)]
            marker = marker_styles[idx % len(marker_styles)]
            
            # Extract data (assuming same structure as before)
            final_ia_list = combinde_list[0]
            final_ie_list = combinde_list[1]
            
            if not final_ia_list or not final_ie_list:
                print(f"Warning: {file_name} doesn't contain expected data structure")
                continue
            
            # Calculate averages
            IA_cre = [round(float(tf.reduce_mean(final_ia_list[i][2], axis=0)), 4) 
                    for i in range(num_iterations)]
            IE_cre = [round(float(tf.reduce_mean(final_ie_list[i][2], axis=0)), 4) 
                    for i in range(num_iterations)]
            
            # Plot average trajectories
            iterations = range(num_iterations)
            ax1.plot(iterations, IA_cre,
                    color=color, linestyle=line_style, linewidth=1.5,
                    marker=marker, markersize=8, label=f'{file_name} - IA')
            ax1.plot(iterations, IE_cre,
                    color=color, linestyle=line_style, linewidth=1.5,
                    marker=marker, markersize=8, alpha=0.6, label=f'{file_name} - IE')
        
        ax1.set_xlabel("Iteration Index", fontsize=12)
        ax1.set_ylabel("Mutual Information (MI)", fontsize=12)
        ax1.grid(True, which='major', alpha=0.4, linestyle='--', linewidth=0.8)
        ax1.grid(True, which='minor', alpha=0.3, linestyle=':', linewidth=0.4)
        ax1.minorticks_on()
        ax1.set_xlim(-0.5, num_iterations - 0.5)
        ax1.set_ylim(0, 1)
        ax1.legend(loc='best', fontsize=10, ncol=2)
        ax1.set_title("Check/Reliable Edges", fontsize=14)
        
        # Right panel: CE/RE edges
        for idx, (file_name, data) in enumerate(loaded_data.items()):
            color = colors[idx % len(colors)]
            line_style = line_styles[idx % len(line_styles)]
            marker = marker_styles[idx % len(marker_styles)]
            
            final_ia_list = combinde_list[0]
            final_ie_list = combinde_list[1]
            
            if not final_ia_list or not final_ie_list:
                continue
            
            # Calculate averages
            IA_cer = [round(float(tf.reduce_mean(final_ia_list[i][3], axis=0)), 4) 
                    for i in range(num_iterations)]
            IE_cer = [round(float(tf.reduce_mean(final_ie_list[i][3], axis=0)), 4) 
                    for i in range(num_iterations)]
            
            # Plot average trajectories
            iterations = range(num_iterations)
            ax2.plot(iterations, IA_cer,
                    color=color, linestyle=line_style, linewidth=1.5,
                    marker=marker, markersize=8, label=f'{file_name} - IA')
            ax2.plot(iterations, IE_cer,
                    color=color, linestyle=line_style, linewidth=1.5,
                    marker=marker, markersize=8, alpha=0.6, label=f'{file_name} - IE')
        
        ax2.set_xlabel("Iteration Index", fontsize=12)
        ax2.set_ylabel("Mutual Information (MI)", fontsize=12)
        ax2.grid(True, which='major', alpha=0.4, linestyle='--', linewidth=0.8)
        ax2.grid(True, which='minor', alpha=0.3, linestyle=':', linewidth=0.4)
        ax2.minorticks_on()
        ax2.set_xlim(-0.5, num_iterations - 0.5)
        ax2.set_ylim(0, 1)
        ax2.legend(loc='best', fontsize=10, ncol=2)
        ax2.set_title("Collapsed/Reliable Edges", fontsize=14)
        
        plt.tight_layout()
        
        # Save or display
        if save_path:
            self._save_figure(fig, save_path, dpi, 'png')
        
        plt.show()
        
        return fig, loaded_data
    
    def plot_density_strips_multi_files2(self, data_dir_pattern, save_path=None, dpi=300, 
                                       figsize=(20, 8), bandwidth=0.1, max_files=5):
        """
        Plot density strips from multiple data files with side-by-side comparison.
        
        Args:
            data_dir_pattern: Pattern to find data files
            save_path: Path to save figure
            dpi: Figure resolution
            figsize: Figure size
            bandwidth: KDE bandwidth
            max_files: Maximum number of files to display (for clarity)
        """
        
        # Find all data files
        if os.path.isdir(data_dir_pattern):
            search_pattern = os.path.join(data_dir_pattern, "*.pkl")
        else:
            search_pattern = data_dir_pattern
        
        data_files = sorted(glob.glob(search_pattern))[:max_files]
        
        if not data_files:
            raise ValueError(f"No data files found: {search_pattern}")
        
        # Load all data
        all_data = {}
        for file_path in data_files:
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            try:
                with open(file_path, 'rb') as f:
                    all_data[file_name] = pickle.load(f)
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
        
        num_files = len(all_data)
        num_iterations = GL.get_map('num_iterations')
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, num_files, figsize=figsize)
        if num_files == 1:
            axes = axes.reshape(2, 1)
        
        # Colors for different edges
        edge_colors = {
            'IA_cr': 'magenta',
            'IE_cr': 'blue',
            'IA_cre': 'red',
            'IE_cre': 'cyan',
            'IA_cer': 'orange',
            'IE_cer': 'green'
        }
        
        # Plot each file
        for col_idx, (file_name, combined_list) in enumerate(all_data.items()):
            final_ia_list = combined_list[0]
            final_ie_list = combined_list[1]
            
            # Top row: CR/RE edges
            ax_top = axes[0, col_idx]
            
            # Calculate averages
            IA_cre = [round(float(tf.reduce_mean(final_ia_list[i][2], axis=0)), 4) 
                     for i in range(num_iterations)]
            IE_cre = [round(float(tf.reduce_mean(final_ie_list[i][2], axis=0)), 4) 
                     for i in range(num_iterations)]
            
            # Plot density strips for each iteration
            for i in range(num_iterations):
                combined_pair_element = self.get_nonzero_pairs_stack(
                    final_ia_list[i][0], 
                    final_ie_list[i][0]
                )
                
                if combined_pair_element.shape[0] > 10:
                    IA_list = combined_pair_element[:, -2]
                    IE_list = combined_pair_element[:, -1]
                    
                    # IA density
                    if len(IA_list) > 1:
                        kde_ia = gaussian_kde(IA_list, bw_method=bandwidth)
                        x_ia = np.linspace(0, 1, 100)
                        y_ia = kde_ia(x_ia)
                        y_ia = y_ia / y_ia.max() * 0.4
                        ax_top.fill_betweenx(x_ia, i - y_ia, i, 
                                            color='magenta', alpha=0.3)
                    
                    # IE density
                    if len(IE_list) > 1:
                        kde_ie = gaussian_kde(IE_list, bw_method=bandwidth)
                        x_ie = np.linspace(0, 1, 100)
                        y_ie = kde_ie(x_ie)
                        y_ie = y_ie / y_ie.max() * 0.4
                        ax_top.fill_betweenx(x_ie, i, i + y_ie,
                                            color='blue', alpha=0.3)
            
            # Plot averages
            iterations = range(num_iterations)
            ax_top.plot(iterations, IA_cre, 'r-', linewidth=2, marker='o', label='IA Avg')
            ax_top.plot(iterations, IE_cre, 'c-', linewidth=2, marker='s', label='IE Avg')
            
            ax_top.set_xlabel("Iteration")
            ax_top.set_ylabel("MI")
            ax_top.set_title(f"{file_name}\nCheck/Reliable Edges")
            ax_top.set_xlim(-0.5, num_iterations - 0.5)
            ax_top.set_ylim(0, 1)
            ax_top.grid(True, alpha=0.3)
            if col_idx == 0:
                ax_top.legend()
            
            # Bottom row: CE/RE edges
            ax_bottom = axes[1, col_idx]
            
            IA_cer = [round(float(tf.reduce_mean(final_ia_list[i][3], axis=0)), 4) 
                     for i in range(num_iterations)]
            IE_cer = [round(float(tf.reduce_mean(final_ie_list[i][3], axis=0)), 4) 
                     for i in range(num_iterations)]
            
            # Plot density strips
            for i in range(num_iterations):
                combined_pair_element = self.get_nonzero_pairs_stack(
                    final_ia_list[i][1], 
                    final_ie_list[i][1]
                )
                
                if combined_pair_element.shape[0] > 10:
                    IA_list = combined_pair_element[:, -2]
                    IE_list = combined_pair_element[:, -1]
                
                if len(IA_list) > 1:
                    kde_ia = gaussian_kde(IA_list, bw_method=bandwidth)
                    x_ia = np.linspace(0, 1, 100)
                    y_ia = kde_ia(x_ia)
                    y_ia = y_ia / y_ia.max() * 0.4
                    ax_bottom.fill_betweenx(x_ia, i - y_ia, i, 
                                           color='magenta', alpha=0.3)
                
                if len(IE_list) > 1:
                    kde_ie = gaussian_kde(IE_list, bw_method=bandwidth)
                    x_ie = np.linspace(0, 1, 100)
                    y_ie = kde_ie(x_ie)
                    y_ie = y_ie / y_ie.max() * 0.4
                    ax_bottom.fill_betweenx(x_ie, i, i + y_ie,
                                           color='blue', alpha=0.3)
            
            # Plot averages
            ax_bottom.plot(iterations, IA_cer, 'r-', linewidth=2, marker='o', label='IA Avg')
            ax_bottom.plot(iterations, IE_cer, 'c-', linewidth=2, marker='s', label='IE Avg')
            
            ax_bottom.set_xlabel("Iteration")
            ax_bottom.set_ylabel("MI")
            ax_bottom.set_title("Collapsed/Reliable Edges")
            ax_bottom.set_xlim(-0.5, num_iterations - 0.5)
            ax_bottom.set_ylim(0, 1)
            ax_bottom.grid(True, alpha=0.3)
            if col_idx == 0:
                ax_bottom.legend()
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path, dpi, 'png')
        
        plt.show()
        return fig, all_data