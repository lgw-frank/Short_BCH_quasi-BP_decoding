import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import globalmap as GL
from scipy.stats import gaussian_kde
#import matplotlib.cm as cm
import os,pickle,re
#import datetime
import glob
from pathlib import Path

class ScatterEXIT:
    """
    Decoder-agnostic Scatter EXIT (S-EXIT) collector
    """
    def __init__(self):
        self.IA = []
        self.IE = []
        
        # 设置全局字体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 微软雅黑
        plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
        plt.rcParams['font.size'] = 16                # 默认字体大小     
        # 可以设置更多参数
        plt.rcParams['axes.labelsize'] = 22
        plt.rcParams['xtick.labelsize'] = 20
        plt.rcParams['ytick.labelsize'] = 20
        plt.rcParams['legend.fontsize'] = 21
    def clear(self):
        self.IA.clear()
        self.IE.clear()
        
    def record_vn(self, llr_extrinsic, bits):
        """
        Record VN extrinsic MI samples
        """
        mi_cr,mi_ce,mi_cre,mi_cer = self.mutual_information_from_llr(llr_extrinsic, bits)
        self.IA.append((mi_cr,mi_ce,mi_cre,mi_cer))

    def record_cn(self, llr_extrinsic, bits):
        """
        Record CN extrinsic MI samples
        """
        mi_cr,mi_ce,mi_cre,mi_cer = self.mutual_information_from_llr(llr_extrinsic, bits)
        self.IE.append((mi_cr,mi_ce,mi_cre,mi_cer))
    #@tf.function
    def mutual_information_from_llr(self,llr, bits,weight=1.):
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
        #tf.print(count_below)
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
   
    def plot_mean_curves_multi_files(self, data_dir_pattern, save_path=None, dpi=800, 
                                      figsize=(10, 6), bandwidth=0.05, max_files=10,
                                      true_fer_asymptotes=None):
        """
        Plot mean MI curves from multiple data files on the same figure.
        FER curves are plotted in a separate figure with log scale.
        
        Parameters:
        -----------
        data_dir_pattern : str
            Directory path or pattern for finding .pkl files
        save_path : str, optional
            Directory to save the output figure
        dpi : int, default=800
            Resolution for saved figure
        figsize : tuple, default=(20, 8)
            Figure size (width, height) in inches
        bandwidth : float, default=0.05
            Bandwidth parameter (currently not used in plotting)
        max_files : int, default=10
            Maximum number of files to process
        true_fer_asymptotes : list of tuple, optional
            List of (SNR_value, true_FER) pairs for plotting horizontal asymptotes
            Only asymptotes matching the loaded SNR values will be plotted.
            e.g., [(2.0, 0.8), (2.5, 0.3), (3.0, 0.1)]
        """
        num_iterations = GL.get_map('num_iterations')
        threshold_MI = GL.get_map('threshold_MI')
        
        # Find all data files matching the pattern
        if os.path.isdir(data_dir_pattern):
            search_pattern = os.path.join(data_dir_pattern, "*.pkl")
        else:
            search_pattern = data_dir_pattern
        
        data_file_list = sorted(glob.glob(search_pattern))[:max_files]
        if not data_file_list:
            raise ValueError(f"No data files found: {search_pattern}")
        
        # Create two figures: one for MI, one for FER
        fig_mi, ax_mi = plt.subplots(figsize=figsize)
        fig_fer, ax_fer = plt.subplots(figsize=figsize)
        
        # Define color and marker styles for different files
        colors = plt.cm.tab10(np.linspace(0, 1, len(data_file_list)))
        markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', '>', '<']
        line_styles = ['--', '-']
        
        # Prepare true_fer_asymptotes if provided
        if true_fer_asymptotes is not None:
            true_fer_dict = dict(true_fer_asymptotes)
        else:
            true_fer_dict = {}
        
        # Track FER curve handles and labels
        fer_curve_handles = []
        fer_curve_labels = []
        
        # Track asymptote handles and labels
        asymptote_handles = []
        asymptote_labels = []
        
        # Store FER data and SNR values for matching
        fer_data = {}  # key: snr_value, value: (FER_estimate, color, label)
        loaded_snr_values = []
        
        for file_idx, data_file in enumerate(data_file_list):
            # Load data from file
            with open(data_file, 'rb') as f:
                final_ia_list, final_ie_list = pickle.load(f)
            
            # Calculate mean MI values
            IA_collapsed_edge_realization = [
                round(float(tf.reduce_mean(final_ia_list[i][2], axis=0)), 4) 
                for i in range(len(final_ia_list))
            ]
            IE_collapsed_edge_realization = [
                round(float(tf.reduce_mean(final_ie_list[i][2], axis=0)), 4) 
                for i in range(len(final_ie_list))
            ]
            
            # Calculate FER estimates
            FER_estimate = [
                self.proportion_below_threshold(
                    tf.reshape(final_ie_list[i][1], [-1]), threshold_MI
                ) 
                for i in range(num_iterations)
            ]
            
            # Print summary statistics
            tf.print(f'File {file_idx+1}/{len(data_file_list)}: {Path(data_file).name}')
            tf.print(f'  IEV: {IA_collapsed_edge_realization}')
            tf.print(f'  IEC: {IE_collapsed_edge_realization}')
            tf.print(f'  FER: {FER_estimate}')
            
            # Extract SNR value from filename
            snr_value = None
            snr_match = re.search(r'(\d+(?:\.\d+)?)[_\s]*dB', data_file, re.IGNORECASE)            
            if snr_match:
                snr_value = float(snr_match.group(1))
                label_prefix = f"{snr_value}dB"
            else:
                # Try to extract other identifiers
                db_match = re.search(r'(\d+)[dB]', data_file)
                if db_match:
                    snr_value = float(db_match.group(1))
                    label_prefix = f"{db_match.group(1)}dB"
                else:
                    label_prefix = f"File{file_idx+1}"
            
            # Select plotting style for this file
            color = colors[file_idx % len(colors)]
            marker = markers[file_idx % len(markers)]
            
            # Store SNR value and FER data
            if snr_value is not None:
                loaded_snr_values.append(snr_value)
                fer_data[snr_value] = (FER_estimate, color, label_prefix)
            else:
                # Use file index as key if no SNR found
                fer_data[file_idx] = (FER_estimate, color, label_prefix)
            
            # Plot IA curve on MI figure
            iterations = range(1, num_iterations+1)
            label_text_mi_ia = rf'{label_prefix} - $I_{{E,V}}$'
            ia_line, = ax_mi.plot(
                iterations, IA_collapsed_edge_realization,
                color=color, linestyle=line_styles[0], linewidth=1.5,
                marker=marker, markersize=6, markevery=max(1, num_iterations//10),
                label=label_text_mi_ia
            )
            
            # Plot IE curve on MI figure
            label_text_mi_ie = rf'{label_prefix} - $I_{{E,C}}$'
            ie_line, = ax_mi.plot(
                iterations, IE_collapsed_edge_realization,
                color=color, linestyle=line_styles[1], linewidth=1.5,
                marker=marker, markersize=6, markevery=max(1, num_iterations//10),
                alpha=0.7, label=label_text_mi_ie
            )
            
            # Plot FER curve on FER figure
            #label_text_fer = f'{label_prefix} (Estimated)'
            label_text_fer = f'{label_prefix}'
            fer_line, = ax_fer.semilogy(
                iterations, FER_estimate,
                color=color, linestyle='-', linewidth=2.0,
                marker=marker, markersize=8, markevery=max(1, num_iterations//10),
                label=label_text_fer
            )
            
            # Store FER curve handles for legend
            fer_curve_handles.append(fer_line)
            fer_curve_labels.append(label_text_fer)
        
        # Plot true FER asymptotes only for matching SNR values
        if true_fer_dict and loaded_snr_values:
            # Find matching SNR values (with tolerance for floating point comparison)
            for snr_value in loaded_snr_values:
                # Look for exact match or close match (within 0.1 dB)
                matching_snr = None
                for true_snr in true_fer_dict.keys():
                    if abs(true_snr - snr_value) < 0.11:  # Allow 0.1 dB tolerance
                        matching_snr = true_snr
                        break
                
                if matching_snr is not None:
                    true_fer_value = true_fer_dict[matching_snr]
                    
                    # Get the corresponding color from the FER curve
                    if snr_value in fer_data:
                        fer_color = fer_data[snr_value][1]
                        
                        # Plot horizontal asymptote line with same color but different style
                        asymptote_line = ax_fer.axhline(
                            y=true_fer_value, 
                            color=fer_color, 
                            linestyle='--', 
                            linewidth=2.5,
                            alpha=0.8,
                            dashes=(6, 4)  # Custom dash pattern
                        )
                        
                        # Add to asymptote handles for legend
                        asymptote_handles.append(asymptote_line)
                        asymptote_labels.append(f'SNR={matching_snr}dB (True FER)')
                        
                        # Add value annotation at the right edge
                        ax_fer.text(
                            #num_iterations * 1.01,  # Slightly beyond x-limit
                            num_iterations * 0.8,  # Slightly less than x-limit
                            true_fer_value*0.7,
                            f'  {true_fer_value:.3f}',
                            verticalalignment='center',
                            fontsize=22,
                            color=fer_color,
                            alpha=0.7
                            #fontweight='bold'
                        )
        
        # Configure MI figure axis properties
        ax_mi.set_xlabel("Iteration index", fontsize=22)
        ax_mi.set_ylabel("Mutual Informaton (MI)", fontsize=22)
        # ax_mi.set_title(
        #     f"MI Evolution Across Iterations",
        #     fontsize=16, pad=20
        # )
        
        # Configure MI figure ticks
        ax_mi.xaxis.set_major_locator(plt.MultipleLocator(2))     
        ax_mi.yaxis.set_major_locator(plt.MultipleLocator(0.1))   
        ax_mi.yaxis.set_minor_locator(plt.MultipleLocator(0.01))  
        ax_mi.tick_params(axis='both', which='both', direction='in',labelsize=20, 
               top=True, right=True, bottom=False, left=True)  
        
        # Configure MI figure grid
        ax_mi.grid(True, which='major', alpha=0.4, linestyle='-.', linewidth=0.8)
        ax_mi.grid(True, which='minor', alpha=0.3, linestyle=':', linewidth=0.4)   
        ax_mi.set_xlim(1, num_iterations)
        ax_mi.set_ylim(0, 1.05)
        
        # Configure MI figure legend
        ax_mi.legend(
            loc='best', fontsize=21, framealpha=0.7,
            ncol=3, columnspacing=0.2, handletextpad=0.2
        )
        
        plt.tight_layout()
        
        # Configure FER figure axis properties (with log scale)
        ax_fer.set_xlabel("Iteration index", fontsize=23)
        ax_fer.set_ylabel("FER", fontsize=23)
        
        # Update title based on whether asymptotes are plotted
        # if asymptote_handles:
        #     ax_fer.set_title(
        #         f"FER Evolution Across Iterations (Log Scale) with True FER Asymptotes",
        #         fontsize=16, pad=20
        #     )
        # else:
        #     ax_fer.set_title(
        #         f"FER Evolution Across Iterations (Log Scale)",
        #         fontsize=16, pad=20
        #     )
        
        # Configure FER figure ticks (log scale)
        ax_fer.xaxis.set_major_locator(plt.MultipleLocator(2))     
        ax_fer.tick_params(axis='both', which='both', direction='in', labelsize=21,
               top=True, right=True, bottom=False, left=True)  
        
        # Configure FER figure grid (log scale compatible)
        ax_fer.grid(True, which='major', alpha=0.4, linestyle='-.', linewidth=0.8)
        ax_fer.grid(True, which='minor', alpha=0.2, linestyle=':', linewidth=0.4)   
        ax_fer.set_xlim(1, num_iterations)
        
        # Set appropriate y-limits for log scale
        all_fer_values = []
        for fer_vals, _, _ in fer_data.values():
            all_fer_values.extend(fer_vals)
        
        # Also consider true FER values for y-limits if asymptotes exist
        if asymptote_handles:
            for snr_value in loaded_snr_values:
                for true_snr, true_fer in true_fer_dict.items():
                    if abs(true_snr - snr_value) < 0.11:
                        all_fer_values.append(true_fer)
        
        min_fer = max(min(all_fer_values), 1e-5)  # Avoid log(0)
        max_fer = max(all_fer_values)
        ax_fer.set_ylim(min_fer * 0.5, max_fer * 2)  # Add some margin
        
        # Create combined legend for FER figure
        all_handles = fer_curve_handles + asymptote_handles
        #all_labels = fer_curve_labels + asymptote_labels
        all_labels = fer_curve_labels 
        
        # Create legend with two columns if there are many entries
        ncol = 2 if len(all_handles) > 6 else 1
        
        ax_fer.legend(
            all_handles, all_labels,
            loc='best', fontsize=21, framealpha=0.7,
            ncol=ncol, columnspacing=0.5, handletextpad=0.5,
            #title='FER Curves (Solid=Estimated, Dashed=True)' if asymptote_handles else 'FER Curves'
            #title='FER Curves'
        )
        
        # Add a note about line styles if asymptotes are present
        # if asymptote_handles:
        #     ax_fer.text(
        #         0.02, 0.02,
        #         'Solid lines: Estimated FER\nDashed lines: True FER asymptotes',
        #         transform=ax_fer.transAxes, fontsize=9,
        #         verticalalignment='bottom',
        #         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7)
        #     )
        
        plt.tight_layout()
        
        # Save figures if save_path is provided
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Generate meaningful filenames
            threshold_MI = GL.get_map('threshold_MI')
            redundancy_factor = GL.get_map('redundancy_factor')
            num_shifts = GL.get_map('num_shifts')
            validate_dataset_size = GL.get_map('validate_dataset_size')
            discriminator_str = f'threshold_{threshold_MI}redundfactor_{redundancy_factor}numshifts_{num_shifts}datasize{validate_dataset_size}'
            
            filename_mi = f"MI_{discriminator_str}.png"
            filename_fer = f"FER_{discriminator_str}.png"
            
            # Add asymptote info to filename if applicable
            if asymptote_handles:
                filename_fer = filename_fer.replace('.png', '_with_asymptotes.png')
            
            # Save MI figure
            full_path_mi = save_path / filename_mi
            fig_mi.savefig(full_path_mi, dpi=dpi, bbox_inches='tight', facecolor='white')
            tf.print(f"MI figure saved to: {full_path_mi}")
            
            # Save FER figure
            full_path_fer = save_path / filename_fer
            fig_fer.savefig(full_path_fer, dpi=dpi, bbox_inches='tight', facecolor='white')
            tf.print(f"FER figure saved to: {full_path_fer}")
        
        # Display plots
        plt.show()
        return fig_mi, fig_fer

    def process_discriminator(self,discriminator_str, key_attribute):
        num_shifts = GL.get_map('num_shifts')
        if key_attribute == 'num_shifts-':
            # Remove numshifts-{num_shifts} part
            substring = f'numshifts-{num_shifts}'
            discriminator_str = discriminator_str.replace(substring, '')
        elif key_attribute == 'num_rows-':
            redundancy_factor = GL.get_map('redundancy_factor')
            # Remove redundfactor-{redundancy_factor} part
            substring = f'redundfactor-{redundancy_factor}'
            discriminator_str = discriminator_str.replace(substring, '')
        
        # Clean up any double dashes or extra separators
        discriminator_str = discriminator_str.replace('--', '-').strip('-')
        return discriminator_str  

    
    def plot_diff_redundancy_shift(self, data_dir_pattern, save_path=None, dpi=800, 
                                   figsize=(10,6.3), bandwidth=0.05, max_files=30,
                                   keystr=None,
                                   # New font size parameters with default values
                                   tick_fontsize=20,
                                   label_fontsize=22,
                                   legend_fontsize=20,
                                   title_fontsize=24):
        """
        Plot the evolution of Mutual Information (IEC) across iterations for different redundancy shifts.
        
        Parameters:
        -----------
        data_dir_pattern : str
            Directory pattern or glob pattern for finding data files
        save_path : str or Path, optional
            Path to save the figures
        dpi : int, default=800
            Resolution for saved figures
        figsize : tuple, default=(20, 8)
            Figure size (width, height)
        bandwidth : float, default=0.05
            Bandwidth parameter (not used in current implementation)
        max_files : int, default=30
            Maximum number of files to process
        keystr : str, optional
            Key string to group data files (e.g., 'alpha', 'beta', etc.)
        tick_fontsize : int, default=12
            Font size for axis tick labels
        label_fontsize : int, default=14
            Font size for axis labels (xlabel, ylabel)
        legend_fontsize : int, default=10
            Font size for legend text
        title_fontsize : int, default=16
            Font size for plot title (if used)
        
        Returns:
        --------
        fig_mi : matplotlib.figure.Figure
            The generated MI figure
        group_snr_iec_data : dict
            Dictionary containing the averaged IEC data for each group and SNR
        """
        num_iterations = GL.get_map('num_iterations')
        iterations = range(1, num_iterations + 1)      
        # Find all data files matching the pattern
        if os.path.isdir(data_dir_pattern):
            search_pattern = os.path.join(data_dir_pattern, "*.pkl")
        else:
            search_pattern = data_dir_pattern
        
        data_file_list = sorted(glob.glob(search_pattern))[:max_files]
        if not data_file_list:
            raise ValueError(f"No data files found: {search_pattern}")     
        # Group files by keystr value and SNR
        residual_str = 'num_shifts-'
        multiplier = 1
        residual_multiplier = 3
        if keystr == 'num_shifts-':
            multiplier = 3
            residual_multiplier = 1
            residual_str = 'num_rows-'
            
        grouped_data = {}    
        for data_file in data_file_list:
            # Extract keystr value
            keystr_value = None
            if residual_str:
                # Pattern to match residual_str followed by digits (e.g., alpha123, beta456)
                keystr_pattern = f'{residual_str}\\d+'
                keystr_match = re.search(keystr_pattern, data_file)
                if keystr_match:
                    value_pattern = r'\d+'
                    value_match = re.search(value_pattern, keystr_match.group())
                    if value_match:
                        keystr_value = float(value_match.group())  # Convert to float for numeric sorting           
            # Extract SNR value
            snr_value = None
            snr_match = re.search(r'(\d+(?:\.\d+)?)[_\s]*dB', data_file, re.IGNORECASE)
            if snr_match:
                snr_value = float(snr_match.group(1))
            else:
                db_match = re.search(r'(\d+)[dB]', data_file)
                if db_match:
                    snr_value = float(db_match.group(1))
            
            if keystr_value is not None and snr_value is not None:
                group_key = keystr_value
                if group_key not in grouped_data:
                    grouped_data[group_key] = {}
                if snr_value not in grouped_data[group_key]:
                    grouped_data[group_key][snr_value] = []
                
                grouped_data[group_key][snr_value].append(data_file)
        
            # Handle case where no files were grouped by keystr
            if not grouped_data:
                print("Warning: No files grouped by keystr. Will process all files individually.")
                for data_file in data_file_list:
                    snr_value = None
                    snr_match = re.search(r'(\d+(?:\.\d+)?)[_\s]*dB', data_file, re.IGNORECASE)
                    if snr_match:
                        snr_value = float(snr_match.group(1))
                        group_key = f"Ungrouped_{snr_value}dB"
                        if group_key not in grouped_data:
                            grouped_data[group_key] = {snr_value: []}
                        grouped_data[group_key][snr_value].append(data_file)
        
        # Sort grouped_data by keystr value (numerical sorting)
        def sort_grouped_data(data_dict):
            """
            Sort the grouped data dictionary by keystr value (numerical order)
            and within each keystr, sort by SNR value.
            """
            sorted_dict = {}
            
            # Sort keystr values
            sorted_keys = []
            for key in data_dict.keys():
                if isinstance(key, (int, float)):
                    sorted_keys.append((key, 'numeric'))
                else:
                    # For string keys, try to extract numeric part
                    num_match = re.search(r'(\d+(?:\.\d+)?)', str(key))
                    if num_match:
                        sorted_keys.append((float(num_match.group(1)), 'string_with_number'))
                    else:
                        sorted_keys.append((float('inf'), 'string'))  # Put non-numeric strings at the end
            
            # Sort by the extracted/numeric value
            sorted_keys.sort(key=lambda x: x[0])
            
            # Reconstruct dictionary with sorted keys
            for key_val, key_type in sorted_keys:
                # Find the original key that matches
                for orig_key in data_dict.keys():
                    if key_type == 'numeric' and orig_key == key_val:
                        sorted_dict[orig_key] = data_dict[orig_key]
                        break
                    elif key_type == 'string_with_number':
                        num_match = re.search(r'(\d+(?:\.\d+)?)', str(orig_key))
                        if num_match and float(num_match.group(1)) == key_val:
                            sorted_dict[orig_key] = data_dict[orig_key]
                            break
            
            # Sort SNR values within each keystr group
            for keystr_val in sorted_dict:
                snr_dict = sorted_dict[keystr_val]
                sorted_snr_items = sorted(snr_dict.items(), key=lambda x: x[0])
                sorted_dict[keystr_val] = dict(sorted_snr_items)
            
            return sorted_dict
        
        # Apply sorting
        grouped_data = sort_grouped_data(grouped_data)
        
        # Create figure for MI
        fig_mi, ax_mi = plt.subplots(figsize=figsize)
        
        # Define color and marker styles
        colors = plt.cm.tab10(np.linspace(0, 1, len(grouped_data)))
        markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', '>', '<']
        line_styles = ['-', '--', '-.', ':']
        
        # Store average IEC values for each group and SNR
        group_snr_iec_data = {}
        
        # Store handles and labels for legend sorting
        legend_handles = []
        legend_labels = []
        
        # Process each group (sorted by keystr value)
        for group_idx, (group_key, snr_files_dict) in enumerate(grouped_data.items()):
            group_snr_iec_data[group_key] = {}
            color = colors[group_idx % len(colors)]
            
            # Process each SNR value (already sorted within the group)
            for snr_idx, (snr_value, file_list) in enumerate(snr_files_dict.items()):
                marker = markers[snr_idx % len(markers)]
                line_style = line_styles[group_idx % len(line_styles)]
                
                all_iec_values = []
                
                # Process each file for this SNR
                for file_idx, data_file in enumerate(file_list):
                    # Load data from file
                    with open(data_file, 'rb') as f:
                        final_ia_list, final_ie_list = pickle.load(f)
                    
                    # Calculate mean MI values
                    IE_collapsed_edge_realization = [
                        round(float(tf.reduce_mean(final_ie_list[i][2], axis=0)), 4) 
                        for i in range(len(final_ie_list))
                    ]
                    
                    all_iec_values.append(IE_collapsed_edge_realization)
                
                # Calculate average IEC across files with same SNR
                if all_iec_values:
                    avg_iec = np.mean(all_iec_values, axis=0)
                    group_snr_iec_data[group_key][snr_value] = avg_iec
                    
                    # Create label based on keystr value and SNR
                    if isinstance(group_key, (int, float)):
                        #label_text = rf'SNR={snr_value}dB,{residual_str[:-1]}={int(group_key)*residual_multiplier}'
                        label_text = rf'({snr_value}dB,{int(group_key)*residual_multiplier})'
                    elif isinstance(group_key, str) and 'Ungrouped' in group_key:
                        # Handle ungrouped case
                        label_text = f'SNR={snr_value}dB'
                    else:
                        label_text = f'SNR={snr_value}dB,{group_key}'
                    
                    # Plot IEC curve
                    line, = ax_mi.plot(
                        iterations, avg_iec,
                        color=color, linestyle=line_style, linewidth=2,
                        marker=marker, markersize=8, markevery=max(1, num_iterations // 10),
                        alpha=0.8, label=label_text
                    )
                    
                    # Store handle and label for later legend sorting
                    legend_handles.append(line)
                    legend_labels.append(label_text)
                    
                    # Add variance visualization (optional)
                    if len(all_iec_values) > 1:
                        std_iec = np.std(all_iec_values, axis=0)
                        ax_mi.fill_between(
                            iterations, 
                            avg_iec - std_iec, 
                            avg_iec + std_iec,
                            alpha=0.2, color=color
                        )
        
        # Configure MI figure axis properties - UPDATED with label_fontsize
        ax_mi.set_xlabel("Iteration Index", fontsize=label_fontsize)
        ax_mi.set_ylabel("Mutual Information (MI)", fontsize=label_fontsize)        
        # Create title with keystr information
        if keystr:
            title = r"Evolution of $I_{E,C}$  across iterations"        
        else:
            title = "$I_{E,C}$ Evolution Across Iterations"
        
        #ax_mi.set_title(title, fontsize=16, pad=20)
        # Configure MI figure ticks - UPDATED with tick_fontsize
        ax_mi.xaxis.set_major_locator(plt.MultipleLocator(2))
        ax_mi.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        ax_mi.yaxis.set_minor_locator(plt.MultipleLocator(0.01))
        ax_mi.tick_params(axis='both', which='both', direction='in', 
                          top=True, right=True, bottom=False, left=True,
                          labelsize=tick_fontsize)  # Added labelsize parameter        
        
        # Configure MI figure grid
        ax_mi.grid(True, which='major', alpha=0.4, linestyle='-.', linewidth=0.8)
        ax_mi.grid(True, which='minor', alpha=0.3, linestyle=':', linewidth=0.4)
        ax_mi.set_xlim(1, num_iterations)
        ax_mi.set_ylim(0, 1.05)
        
        # Sort legend by keystr value and then SNR
        def extract_sort_keys(label):
            """
            Extract sorting keys from legend label.
            Returns (keystr_value, snr_value) for sorting.
            """
            # Initialize with default values (for ungrouped cases)
            keystr_val = float('inf')
            snr_val = float('inf')
            
            # Try to extract keystr value
            if residual_str:
                keystr_pattern = rf'{residual_str[:-1]}=([\d.]+)'
                keystr_match = re.search(keystr_pattern, label)
                if keystr_match:
                    keystr_val = float(keystr_match.group(1))
            
            # Try to extract SNR value
            snr_pattern = r'SNR=([\d.]+)dB'
            snr_match = re.search(snr_pattern, label)
            if snr_match:
                snr_val = float(snr_match.group(1))
            
            return (keystr_val, snr_val)
        
        # Sort handles and labels
        if residual_str:
            # Sort by keystr value, then by SNR
            sorted_indices = sorted(
                range(len(legend_labels)), 
                key=lambda i: extract_sort_keys(legend_labels[i])
            )
            sorted_handles = [legend_handles[i] for i in sorted_indices]
            sorted_labels = [legend_labels[i] for i in sorted_indices]
        else:
            # If no keystr, sort by SNR only
            sorted_handles = legend_handles
            sorted_labels = legend_labels
        
        # Configure MI figure legend - UPDATED with legend_fontsize
        ax_mi.legend(
            sorted_handles, sorted_labels,
            loc='best', fontsize=legend_fontsize, framealpha=0.7,
            ncol=3, columnspacing=0.2, handletextpad=0.5
        )        
        plt.tight_layout()
        
        # Save figures if save_path is provided
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Generate meaningful filenames
            num_shifts = GL.get_map('num_shifts')
            validate_dataset_size = GL.get_map('validate_dataset_size')
            code = GL.get_map('code_parameters')
            
            # Include keystr in filename if provided
            if keystr == 'num_shifts-':
                keystr_str = f"{keystr}{num_shifts*multiplier}grouped-"
            elif keystr == 'num_rows-':
                keystr_str = f"{keystr}{code.H.shape[0]}grouped-"    
            else:
                keystr_str = ""
            
            discriminator_str = f'{keystr_str}datasize-{validate_dataset_size}'
            #residual_string = self.process_discriminator(discriminator_str,keystr)
            filename_mi = f"MI_{discriminator_str}.png"
            
            # Save MI figure
            full_path_mi = save_path / filename_mi
            fig_mi.savefig(full_path_mi, dpi=dpi, bbox_inches='tight', facecolor='white')
            tf.print(f"MI figure saved to: {full_path_mi}")
            
            # Optional: Save data for further analysis
            data_filename = f"IEC_data_{discriminator_str}.pkl"
            data_path = save_path / data_filename
            with open(data_path, 'wb') as f:
                pickle.dump(group_snr_iec_data, f)
            tf.print(f"IEC data saved to: {data_path}")
        
        # Display plot
        plt.show()
        
        return fig_mi, group_snr_iec_data
    
    def plot_diff_redundancy_shift2(self, data_dir_pattern,save_path=None, dpi=800, 
                                 figsize=(20, 8), bandwidth=0.05, max_files=30,
                                 keystr=None):
        """
        Plot the evolution of Mutual Information (IEC) across iterations for different redundancy shifts.
        
        Parameters:
        -----------
        data_dir_pattern : str
            Directory pattern or glob pattern for finding data files
        save_path : str or Path, optional
            Path to save the figures
        dpi : int, default=800
            Resolution for saved figures
        figsize : tuple, default=(20, 8)
            Figure size (width, height)
        bandwidth : float, default=0.05
            Bandwidth parameter (not used in current implementation)
        max_files : int, default=30
            Maximum number of files to process
        keystr : str, optional
            Key string to group data files (e.g., 'alpha', 'beta', etc.)
        
        Returns:
        --------
        fig_mi : matplotlib.figure.Figure
            The generated MI figure
        group_snr_iec_data : dict
            Dictionary containing the averaged IEC data for each group and SNR
        """
        num_iterations = GL.get_map('num_iterations')
        iterations = range(1, num_iterations + 1)      
        # Find all data files matching the pattern
        if os.path.isdir(data_dir_pattern):
            search_pattern = os.path.join(data_dir_pattern, "*.pkl")
        else:
            search_pattern = data_dir_pattern
        
        data_file_list = sorted(glob.glob(search_pattern))[:max_files]
        if not data_file_list:
            raise ValueError(f"No data files found: {search_pattern}")     
        # Group files by keystr value and SNR
        residual_str = 'num_shifts-'
        multiplier = 1
        residual_multiplier = 3
        if keystr == 'num_shifts-':
            multiplier = 3
            residual_multiplier = 1
            residual_str = 'num_rows-'
            
        grouped_data = {}    
        for data_file in data_file_list:
            # Extract keystr value
            keystr_value = None
            if residual_str:
                # Pattern to match residual_str followed by digits (e.g., alpha123, beta456)
                keystr_pattern = f'{residual_str}\\d+'
                keystr_match = re.search(keystr_pattern, data_file)
                if keystr_match:
                    value_pattern = r'\d+'
                    value_match = re.search(value_pattern, keystr_match.group())
                    if value_match:
                        keystr_value = float(value_match.group())  # Convert to float for numeric sorting           
            # Extract SNR value
            snr_value = None
            snr_match = re.search(r'(\d+(?:\.\d+)?)[_\s]*dB', data_file, re.IGNORECASE)
            if snr_match:
                snr_value = float(snr_match.group(1))
            else:
                db_match = re.search(r'(\d+)[dB]', data_file)
                if db_match:
                    snr_value = float(db_match.group(1))
            
            if keystr_value is not None and snr_value is not None:
                group_key = keystr_value
                if group_key not in grouped_data:
                    grouped_data[group_key] = {}
                if snr_value not in grouped_data[group_key]:
                    grouped_data[group_key][snr_value] = []
                
                grouped_data[group_key][snr_value].append(data_file)
        
            # Handle case where no files were grouped by keystr
            if not grouped_data:
                print("Warning: No files grouped by keystr. Will process all files individually.")
                for data_file in data_file_list:
                    snr_value = None
                    snr_match = re.search(r'(\d+(?:\.\d+)?)[_\s]*dB', data_file, re.IGNORECASE)
                    if snr_match:
                        snr_value = float(snr_match.group(1))
                        group_key = f"Ungrouped_{snr_value}dB"
                        if group_key not in grouped_data:
                            grouped_data[group_key] = {snr_value: []}
                        grouped_data[group_key][snr_value].append(data_file)
        
        # Sort grouped_data by keystr value (numerical sorting)
        def sort_grouped_data(data_dict):
            """
            Sort the grouped data dictionary by keystr value (numerical order)
            and within each keystr, sort by SNR value.
            """
            sorted_dict = {}
            
            # Sort keystr values
            sorted_keys = []
            for key in data_dict.keys():
                if isinstance(key, (int, float)):
                    sorted_keys.append((key, 'numeric'))
                else:
                    # For string keys, try to extract numeric part
                    num_match = re.search(r'(\d+(?:\.\d+)?)', str(key))
                    if num_match:
                        sorted_keys.append((float(num_match.group(1)), 'string_with_number'))
                    else:
                        sorted_keys.append((float('inf'), 'string'))  # Put non-numeric strings at the end
            
            # Sort by the extracted/numeric value
            sorted_keys.sort(key=lambda x: x[0])
            
            # Reconstruct dictionary with sorted keys
            for key_val, key_type in sorted_keys:
                # Find the original key that matches
                for orig_key in data_dict.keys():
                    if key_type == 'numeric' and orig_key == key_val:
                        sorted_dict[orig_key] = data_dict[orig_key]
                        break
                    elif key_type == 'string_with_number':
                        num_match = re.search(r'(\d+(?:\.\d+)?)', str(orig_key))
                        if num_match and float(num_match.group(1)) == key_val:
                            sorted_dict[orig_key] = data_dict[orig_key]
                            break
            
            # Sort SNR values within each keystr group
            for keystr_val in sorted_dict:
                snr_dict = sorted_dict[keystr_val]
                sorted_snr_items = sorted(snr_dict.items(), key=lambda x: x[0])
                sorted_dict[keystr_val] = dict(sorted_snr_items)
            
            return sorted_dict
        
        # Apply sorting
        grouped_data = sort_grouped_data(grouped_data)
        
        # Create figure for MI
        fig_mi, ax_mi = plt.subplots(figsize=figsize)
        
        # Define color and marker styles
        colors = plt.cm.tab10(np.linspace(0, 1, len(grouped_data)))
        markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', '>', '<']
        line_styles = ['-', '--', '-.', ':']
        
        # Store average IEC values for each group and SNR
        group_snr_iec_data = {}
        
        # Store handles and labels for legend sorting
        legend_handles = []
        legend_labels = []
        
        # Process each group (sorted by keystr value)
        for group_idx, (group_key, snr_files_dict) in enumerate(grouped_data.items()):
            group_snr_iec_data[group_key] = {}
            color = colors[group_idx % len(colors)]
            
            # Process each SNR value (already sorted within the group)
            for snr_idx, (snr_value, file_list) in enumerate(snr_files_dict.items()):
                marker = markers[snr_idx % len(markers)]
                line_style = line_styles[group_idx % len(line_styles)]
                
                all_iec_values = []
                
                # Process each file for this SNR
                for file_idx, data_file in enumerate(file_list):
                    # Load data from file
                    with open(data_file, 'rb') as f:
                        final_ia_list, final_ie_list = pickle.load(f)
                    
                    # Calculate mean MI values
                    IE_collapsed_edge_realization = [
                        round(float(tf.reduce_mean(final_ie_list[i][2], axis=0)), 4) 
                        for i in range(len(final_ie_list))
                    ]
                    
                    all_iec_values.append(IE_collapsed_edge_realization)
                
                # Calculate average IEC across files with same SNR
                if all_iec_values:
                    avg_iec = np.mean(all_iec_values, axis=0)
                    group_snr_iec_data[group_key][snr_value] = avg_iec
                    
                    # Create label based on keystr value and SNR
                    if isinstance(group_key, (int, float)):
                        label_text = rf'SNR={snr_value}dB,{residual_str[:-1]}={int(group_key)*residual_multiplier}'
                    elif isinstance(group_key, str) and 'Ungrouped' in group_key:
                        # Handle ungrouped case
                        label_text = f'SNR={snr_value}dB'
                    else:
                        label_text = f'SNR={snr_value}dB,{group_key}'
                    
                    # Plot IEC curve
                    line, = ax_mi.plot(
                        iterations, avg_iec,
                        color=color, linestyle=line_style, linewidth=2,
                        marker=marker, markersize=8, markevery=max(1, num_iterations // 10),
                        alpha=0.8, label=label_text
                    )
                    
                    # Store handle and label for later legend sorting
                    legend_handles.append(line)
                    legend_labels.append(label_text)
                    
                    # Add variance visualization (optional)
                    if len(all_iec_values) > 1:
                        std_iec = np.std(all_iec_values, axis=0)
                        ax_mi.fill_between(
                            iterations, 
                            avg_iec - std_iec, 
                            avg_iec + std_iec,
                            alpha=0.2, color=color
                        )
        
        # Configure MI figure axis properties
        ax_mi.set_xlabel("Iteration Index", fontsize=16)
        ax_mi.set_ylabel("Mutual Information (MI)", fontsize=16)
        
        # Create title with keystr information
        # if keystr:
        #     title = r"IEC Evolution Across Iterations (Grouped by number of shifts in inputs)"        
        # else:
        #     title = "IEC Evolution Across Iterations"
        
        # ax_mi.set_title(title, fontsize=16, pad=20)
        
        # Configure MI figure ticks
        ax_mi.xaxis.set_major_locator(plt.MultipleLocator(2))
        ax_mi.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        ax_mi.yaxis.set_minor_locator(plt.MultipleLocator(0.01))
        ax_mi.tick_params(axis='both', which='both', direction='in', labelsize=14,
                          top=True, right=True, bottom=False, left=True)
        
        # Configure MI figure grid
        ax_mi.grid(True, which='major', alpha=0.4, linestyle='-.', linewidth=0.8)
        ax_mi.grid(True, which='minor', alpha=0.3, linestyle=':', linewidth=0.4)
        ax_mi.set_xlim(1, num_iterations)
        ax_mi.set_ylim(0, 1.05)
        
        # Sort legend by keystr value and then SNR
        def extract_sort_keys(label):
            """
            Extract sorting keys from legend label.
            Returns (keystr_value, snr_value) for sorting.
            """
            # Initialize with default values (for ungrouped cases)
            keystr_val = float('inf')
            snr_val = float('inf')
            
            # Try to extract keystr value
            if residual_str:
                keystr_pattern = rf'{residual_str[:-1]}=([\d.]+)'
                keystr_match = re.search(keystr_pattern, label)
                if keystr_match:
                    keystr_val = float(keystr_match.group(1))
            
            # Try to extract SNR value
            snr_pattern = r'SNR=([\d.]+)dB'
            snr_match = re.search(snr_pattern, label)
            if snr_match:
                snr_val = float(snr_match.group(1))
            
            return (keystr_val, snr_val)
        
        # Sort handles and labels
        if residual_str:
            # Sort by keystr value, then by SNR
            sorted_indices = sorted(
                range(len(legend_labels)), 
                key=lambda i: extract_sort_keys(legend_labels[i])
            )
            sorted_handles = [legend_handles[i] for i in sorted_indices]
            sorted_labels = [legend_labels[i] for i in sorted_indices]
        else:
            # If no keystr, sort by SNR only
            sorted_handles = legend_handles
            sorted_labels = legend_labels
        
        # Configure MI figure legend with sorted entries
        ax_mi.legend(
            sorted_handles, sorted_labels,
            loc='best', fontsize=14, framealpha=0.9,
            ncol=2, columnspacing=1.0, handletextpad=0.5
        )
        
        plt.tight_layout()
        
        # Save figures if save_path is provided
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Generate meaningful filenames
            num_shifts = GL.get_map('num_shifts')
            validate_dataset_size = GL.get_map('validate_dataset_size')
            code = GL.get_map('code_parameters')
            
            # Include keystr in filename if provided
            if keystr == 'num_shifts-':
                keystr_str = f"{keystr}{num_shifts*multiplier}grouped-"
            elif keystr == 'num_rows-':
                keystr_str = f"{keystr}{code.H.shape[0]}grouped-"    
            else:
                keystr_str = ""
            
            discriminator_str = f'{keystr_str}datasize-{validate_dataset_size}'
            #residual_string = self.process_discriminator(discriminator_str,keystr)
            filename_mi = f"MI_{discriminator_str}.png"
            
            # Save MI figure
            full_path_mi = save_path / filename_mi
            fig_mi.savefig(full_path_mi, dpi=dpi, bbox_inches='tight', facecolor='white')
            tf.print(f"MI figure saved to: {full_path_mi}")
            
            # Optional: Save data for further analysis
            data_filename = f"IEC_data_{discriminator_str}.pkl"
            data_path = save_path / data_filename
            with open(data_path, 'wb') as f:
                pickle.dump(group_snr_iec_data, f)
            tf.print(f"IEC data saved to: {data_path}")
        
        # Display plot
        plt.show()
        
        return fig_mi, group_snr_iec_data
    
    def plot_density_strips_regular_file(self, data_dir_pattern, save_path=None, dpi=800, 
                                       figsize=(20, 8), bandwidth=0.05, max_files=10):
        
        num_iterations = GL.get_map('num_iterations')
        threshold_MI = GL.get_map('threshold_MI')
        # Find all data files
        if os.path.isdir(data_dir_pattern):
            search_pattern = os.path.join(data_dir_pattern, "*.pkl")
        else:
            search_pattern = data_dir_pattern
        
        data_file_list = sorted(glob.glob(search_pattern))[:max_files]
        
        if not data_file_list:
            raise ValueError(f"No data files found: {search_pattern}")
        for data_file in data_file_list:
            with open(data_file, 'rb') as f:
                final_ia_list,final_ie_list = pickle.load(f)
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
            jitter_strength = 1e-6  # Small jitter to avoid singular matrices
            
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
                    
                    # Kernel Density Estimation for IA with accumulation at 0
                    if len(IA_list) > 1:
                        try:
                            # Calculate KDE for IA with clipping to [0, 1]
                            IA_list_clipped = np.clip(IA_list, 0, 1)
                            
                            # Calculate weight for values below 0
                            below_zero_count = np.sum(IA_list < 0)
                            total_count = len(IA_list)
                            below_zero_weight = below_zero_count / total_count if total_count > 0 else 0
                            
                            # KDE for values in [0, 1]
                            mask_01 = (IA_list_clipped >= 0) & (IA_list_clipped <= 1)
                            data_01 = IA_list_clipped[mask_01]
                            
                            if len(data_01) > 1:
                                # Check if data has variance
                                if np.std(data_01) < 1e-10:  # Almost constant data
                                    # Handle constant data case
                                    x_ia = np.linspace(0, 1, 100)
                                    y_ia = np.zeros_like(x_ia)
                                    # Find the constant value
                                    const_val = data_01[0]
                                    # Find closest index to constant value
                                    const_idx = np.argmin(np.abs(x_ia - const_val))
                                    # Place all probability at that point
                                    y_ia[const_idx] = 1.0 / (x_ia[1] - x_ia[0])  # Convert to density
                                    
                                    # Adjust for below-zero weight
                                    y_ia = y_ia * (1 - below_zero_weight)
                                    
                                    # Add accumulated probability at zero if needed
                                    if below_zero_count > 0:
                                        zero_idx = np.argmin(np.abs(x_ia - 0))
                                        y_ia[zero_idx] += below_zero_weight / (x_ia[1] - x_ia[0])
                                else:
                                    # Add small jitter to avoid singular matrices
                                    data_01_jittered = data_01 + np.random.normal(0, jitter_strength, len(data_01))
                                    kde_ia = gaussian_kde(data_01_jittered, bw_method=bandwidth)
                                    x_ia = np.linspace(0, 1, 100)
                                    y_ia = kde_ia(x_ia)
                                    
                                    # Normalize to maintain total probability = 1
                                    integral = np.trapz(y_ia, x_ia)
                                    if integral > epsilon:
                                        y_ia = y_ia / integral * (1 - below_zero_weight)
                                    else:
                                        y_ia = np.zeros_like(x_ia)
                                    
                                    # Add the accumulated probability at exactly 0
                                    if below_zero_count > 0:
                                        zero_idx = np.argmin(np.abs(x_ia - 0))
                                        y_ia[zero_idx] += below_zero_weight / (x_ia[1] - x_ia[0])
                                
                                # Normalize to width 0.4 for visualization
                                if y_ia.max() > epsilon:
                                    y_ia = y_ia / y_ia.max() * 0.4
                                
                                # Fill IA density strip (left side of iteration)
                                ax1.fill_betweenx(x_ia, i - y_ia, i, 
                                                 color='magenta', alpha=0.5, edgecolor='darkmagenta',
                                                 label='$I_{v2c}$ Distribution' if i == 0 else "")
                        except Exception as e:
                            print(f"Warning: KDE failed for IA at iteration {i}: {e}")
                            # Fallback: simple histogram-based representation
                            x_ia = np.linspace(0, 1, 100)
                            y_ia = np.zeros_like(x_ia)
                            ax1.fill_betweenx(x_ia, i - y_ia, i, 
                                             color='magenta', alpha=0.2, edgecolor='darkmagenta')
                    
                    # Kernel Density Estimation for IE with accumulation at 0
                    if len(IE_list) > 1:
                        try:
                            # Calculate KDE for IE with clipping to [0, 1]
                            IE_list_clipped = np.clip(IE_list, 0, 1)
                            
                            # Calculate weight for values below 0
                            below_zero_count = np.sum(IE_list < 0)
                            total_count = len(IE_list)
                            below_zero_weight = below_zero_count / total_count if total_count > 0 else 0
                            
                            # KDE for values in [0, 1]
                            mask_01 = (IE_list_clipped >= 0) & (IE_list_clipped <= 1)
                            data_01 = IE_list_clipped[mask_01]
                            
                            if len(data_01) > 1:
                                # Check if data has variance
                                if np.std(data_01) < 1e-10:  # Almost constant data
                                    # Handle constant data case
                                    x_ie = np.linspace(0, 1, 100)
                                    y_ie = np.zeros_like(x_ie)
                                    # Find the constant value
                                    const_val = data_01[0]
                                    # Find closest index to constant value
                                    const_idx = np.argmin(np.abs(x_ie - const_val))
                                    # Place all probability at that point
                                    y_ie[const_idx] = 1.0 / (x_ie[1] - x_ie[0])
                                    
                                    # Adjust for below-zero weight
                                    y_ie = y_ie * (1 - below_zero_weight)
                                    
                                    # Add accumulated probability at zero if needed
                                    if below_zero_count > 0:
                                        zero_idx = np.argmin(np.abs(x_ie - 0))
                                        y_ie[zero_idx] += below_zero_weight / (x_ie[1] - x_ie[0])
                                else:
                                    # Add small jitter to avoid singular matrices
                                    data_01_jittered = data_01 + np.random.normal(0, jitter_strength, len(data_01))
                                    kde_ie = gaussian_kde(data_01_jittered, bw_method=bandwidth)
                                    x_ie = np.linspace(0, 1, 100)
                                    y_ie = kde_ie(x_ie)
                                    
                                    # Normalize to maintain total probability = 1
                                    integral = np.trapz(y_ie, x_ie)
                                    if integral > epsilon:
                                        y_ie = y_ie / integral * (1 - below_zero_weight)
                                    else:
                                        y_ie = np.zeros_like(x_ie)
                                    
                                    # Add the accumulated probability at exactly 0
                                    if below_zero_count > 0:
                                        zero_idx = np.argmin(np.abs(x_ie - 0))
                                        y_ie[zero_idx] += below_zero_weight / (x_ie[1] - x_ie[0])
                                
                                # Normalize to width 0.4 for visualization
                                if y_ie.max() > epsilon:
                                    y_ie = y_ie / y_ie.max() * 0.4
                                
                                # Fill IE density strip (right side of iteration)
                                ax1.fill_betweenx(x_ie, i, i + y_ie,
                                                 color='blue', alpha=0.5, edgecolor='darkblue',
                                                 label='$I_{c2v}$ Distribution' if i == 0 else "")
                        except Exception as e:
                            print(f"Warning: KDE failed for IE at iteration {i}: {e}")
                            # Fallback: simple histogram-based representation
                            x_ie = np.linspace(0, 1, 100)
                            y_ie = np.zeros_like(x_ie)
                            ax1.fill_betweenx(x_ie, i, i + y_ie,
                                             color='blue', alpha=0.2, edgecolor='darkblue')
            
            # Plot average trajectories
            iterations = range(num_iterations)
            ax1.plot(iterations, np.clip(IA_cre, 0, 1),
                     'r-', linewidth=1.5, marker='o', markersize=8, label='IA Average')
            ax1.plot(iterations, np.clip(IE_cre, 0, 1),
                     'c-', linewidth=1.5, marker='s', markersize=8, label='IE Average')
            
            ax1.set_xlabel("Iteration Index", fontsize=12)
            ax1.set_ylabel("Mutual Information (MI)", fontsize=12)
            ax1.grid(True, which='major', alpha=0.4, linestyle='--', linewidth=0.8)
            ax1.grid(True, which='minor', alpha=0.3, linestyle=':', linewidth=0.4)
            ax1.minorticks_on()
            ax1.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
            ax1.set_xlim(-0.5, num_iterations - 0.5)
            ax1.set_ylim(0, 1)
            ax1.legend(loc=(0.4,0.05), fontsize=12)
    
            # Right panel: collapsed edges visualization
            for i in range(num_iterations):
                IA_list = IA_ce[i]
                IE_list = IE_ce[i]
                
                # Kernel Density Estimation for IA with accumulation at 0
                if len(IA_list) > 1:
                    try:
                        # Calculate KDE for IA with clipping to [0, 1]
                        IA_list_clipped = np.clip(IA_list, 0, 1)
                        
                        # Calculate weight for values below 0
                        below_zero_count = np.sum(IA_list < 0)
                        total_count = len(IA_list)
                        below_zero_weight = below_zero_count / total_count if total_count > 0 else 0
                        
                        # KDE for values in [0, 1]
                        mask_01 = (IA_list_clipped >= 0) & (IA_list_clipped <= 1)
                        data_01 = IA_list_clipped[mask_01]
                        
                        if len(data_01) > 1:
                            # Check if data has variance
                            if np.std(data_01) < 1e-10:  # Almost constant data
                                # Handle constant data case
                                x_ia = np.linspace(0, 1, 100)
                                y_ia = np.zeros_like(x_ia)
                                # Find the constant value
                                const_val = data_01[0]
                                # Find closest index to constant value
                                const_idx = np.argmin(np.abs(x_ia - const_val))
                                # Place all probability at that point
                                y_ia[const_idx] = 1.0 / (x_ia[1] - x_ia[0])
                                
                                # Adjust for below-zero weight
                                y_ia = y_ia * (1 - below_zero_weight)
                                
                                # Add accumulated probability at zero if needed
                                if below_zero_count > 0:
                                    zero_idx = np.argmin(np.abs(x_ia - 0))
                                    y_ia[zero_idx] += below_zero_weight / (x_ia[1] - x_ia[0])
                            else:
                                # Add small jitter to avoid singular matrices
                                data_01_jittered = data_01 + np.random.normal(0, jitter_strength, len(data_01))
                                kde_ia = gaussian_kde(data_01_jittered, bw_method=bandwidth)
                                x_ia = np.linspace(0, 1, 100)
                                y_ia = kde_ia(x_ia)
                                
                                # Normalize to maintain total probability = 1
                                integral = np.trapz(y_ia, x_ia)
                                if integral > epsilon:
                                    y_ia = y_ia / integral * (1 - below_zero_weight)
                                else:
                                    y_ia = np.zeros_like(x_ia)
                                
                                # Add the accumulated probability at exactly 0
                                if below_zero_count > 0:
                                    zero_idx = np.argmin(np.abs(x_ia - 0))
                                    y_ia[zero_idx] += below_zero_weight / (x_ia[1] - x_ia[0])
                            
                            # Normalize to width 0.4 for visualization
                            if y_ia.max() > epsilon:
                                y_ia = y_ia / y_ia.max() * 0.4
                            
                            # Fill IA density strip (left side of iteration)
                            ax2.fill_betweenx(x_ia, i - y_ia, i, 
                                             color='magenta', alpha=0.5, edgecolor='darkmagenta',
                                             label='$I_{v2c}$ Distribution' if i == 0 else "")
                    except Exception as e:
                        print(f"Warning: KDE failed for IA (right panel) at iteration {i}: {e}")
                        # Fallback: simple histogram-based representation
                        x_ia = np.linspace(0, 1, 100)
                        y_ia = np.zeros_like(x_ia)
                        ax2.fill_betweenx(x_ia, i - y_ia, i, 
                                         color='magenta', alpha=0.2, edgecolor='darkmagenta')
                
                # Kernel Density Estimation for IE with accumulation at 0
                if len(IE_list) > 1:
                    try:
                        # Calculate KDE for IE with clipping to [0, 1]
                        IE_list_clipped = np.clip(IE_list, 0, 1)
                        
                        # Calculate weight for values below 0
                        below_zero_count = np.sum(IE_list < 0)
                        total_count = len(IE_list)
                        below_zero_weight = below_zero_count / total_count if total_count > 0 else 0
                        
                        # KDE for values in [0, 1]
                        mask_01 = (IE_list_clipped >= 0) & (IE_list_clipped <= 1)
                        data_01 = IE_list_clipped[mask_01]
                        
                        if len(data_01) > 1:
                            # Check if data has variance
                            if np.std(data_01) < 1e-10:  # Almost constant data
                                # Handle constant data case
                                x_ie = np.linspace(0, 1, 100)
                                y_ie = np.zeros_like(x_ie)
                                # Find the constant value
                                const_val = data_01[0]
                                # Find closest index to constant value
                                const_idx = np.argmin(np.abs(x_ie - const_val))
                                # Place all probability at that point
                                y_ie[const_idx] = 1.0 / (x_ie[1] - x_ie[0])
                                
                                # Adjust for below-zero weight
                                y_ie = y_ie * (1 - below_zero_weight)
                                
                                # Add accumulated probability at zero if needed
                                if below_zero_count > 0:
                                    zero_idx = np.argmin(np.abs(x_ie - 0))
                                    y_ie[zero_idx] += below_zero_weight / (x_ie[1] - x_ie[0])
                            else:
                                # Add small jitter to avoid singular matrices
                                data_01_jittered = data_01 + np.random.normal(0, jitter_strength, len(data_01))
                                kde_ie = gaussian_kde(data_01_jittered, bw_method=bandwidth)
                                x_ie = np.linspace(0, 1, 100)
                                y_ie = kde_ie(x_ie)
                                
                                # Normalize to maintain total probability = 1
                                integral = np.trapz(y_ie, x_ie)
                                if integral > epsilon:
                                    y_ie = y_ie / integral * (1 - below_zero_weight)
                                else:
                                    y_ie = np.zeros_like(x_ie)
                                
                                # Add the accumulated probability at exactly 0
                                if below_zero_count > 0:
                                    zero_idx = np.argmin(np.abs(x_ie - 0))
                                    y_ie[zero_idx] += below_zero_weight / (x_ie[1] - x_ie[0])
                            
                            # Normalize to width 0.4 for visualization
                            if y_ie.max() > epsilon:
                                y_ie = y_ie / y_ie.max() * 0.4
                            
                            # Fill IE density strip (right side of iteration)
                            ax2.fill_betweenx(x_ie, i, i + y_ie,
                                             color='blue', alpha=0.5, edgecolor='darkblue',
                                             label='$I_{c2v}$ Distribution' if i == 0 else "")
                    except Exception as e:
                        print(f"Warning: KDE failed for IE (right panel) at iteration {i}: {e}")
                        # Fallback: simple histogram-based representation
                        x_ie = np.linspace(0, 1, 100)
                        y_ie = np.zeros_like(x_ie)
                        ax2.fill_betweenx(x_ie, i, i + y_ie,
                                         color='blue', alpha=0.2, edgecolor='darkblue')
            
            # Plot average trajectories
            ax2.plot(iterations, np.clip(IA_cer, 0, 1),
                     'r-', linewidth=1.5, marker='o', markersize=8, label='IA Average')
            ax2.plot(iterations, np.clip(IE_cer, 0, 1),
                     'c-', linewidth=1.5, marker='s', markersize=8, label='IE Average')
            
            ax2.set_xlabel("Iteration Index", fontsize=12)
            ax2.set_ylabel("Mutual Information (MI)", fontsize=12)
            ax2.grid(True, which='major', alpha=0.4, linestyle='--', linewidth=0.8)
            ax2.grid(True, which='minor', alpha=0.3, linestyle=':', linewidth=0.4)
            ax2.minorticks_on()
            ax2.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
            ax2.set_xlim(-0.5, num_iterations - 0.5)
            ax2.set_ylim(0, 1)
            ax2.legend(loc=(0.4,0.05),fontsize=12)        
            plt.tight_layout()

            # Convert to Path object and resolve backslashes
            path = Path(data_file)
            # Extract parts
            snr = path.parent.name  # Gets '2.0dB'
            iteration_part = path.stem  # Gets 'iteration-18_num_shifts-3-sample_size-3000-num_rows-94'           
            # Connect them
            filename = f"{snr}_{iteration_part}"            
            # Save or 
            # Connect save_path and connected_name to form full path
            full_save_path = os.path.join(save_path, filename)
            if save_path:
                self._save_figure(fig, full_save_path, dpi)
            plt.show()
   
    def _save_figure(self, fig, full_save_path,dpi=500):       
        # Save figure
        full_save_path += '.png'
        fig.savefig(full_save_path, dpi=dpi, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        print(f"Figure saved to: {full_save_path} (dpi={dpi})")

   