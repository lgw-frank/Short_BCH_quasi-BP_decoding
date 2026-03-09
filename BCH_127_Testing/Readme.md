# BCH (127,64) Testing Guide

## Development Environment

- **IDE**: Spyder (Anaconda Distribution)
- **OS**: Windows 10
- **Entry Point**: `BCH_127_testing.py`

---
## Configuration Setup
### Command Line Argument Format
In the entry file `BCH_127_testing.py`, configure the parameters as follows:

```python
sys.argv = "python 2.0 4.5 6 100 1000 18 BCH_127_64_10_strip.alist QBP/NMS-1".split()
```
sys.argv = "python <min_snr> <max_snr> <num_points> <batch_size> <num_batches> <max_iterations> <parity_check_matrix_file> <decoder_type>".split()
### Parameter Description
| Parameter | Description | Example |
|-----------|-------------|---------|
| `min_snr` | Minimum SNR value in dB | `2.0` |
| `max_snr` | Maximum SNR value in dB | `4.5` |
| `num_points` | Number of evenly distributed SNR points between min_snr and max_snr | `6` |
| `batch_size` | Number of samples per batch | `100` |
| `num_batches` | Total number of batches to process | `1000` |
| `max_iterations` | Maximum iterations for NMS decoding (higher values yield diminishing FER gains) | `18` |
| `parity_check_matrix_file` | BCH parity-check matrix file | `BCH_127_64_10_strip.alist` |
| `decoder_type` | BP variant to evaluate (`QBP` for conventional BP, `NMS-1` for conventional NMS) | `QBP/NMS-1` |

### Settings Overview (`globalmap.py`)

```python
def global_setting(argv):
    # Matrix configuration
    set_map('regular_matrix', False)                          # Disable conventional parity-check matrix
    set_map('generate_extended_parity_check_matrix', True)    # Enable optimized matrix with redundant rows
    set_map('reduction_iteration', 4)                         # Iterations to acquire minimal-weight rows
    set_map('redundancy_factor', 2)                           # Controls matrix row count (1→63, 1.5→94, 2→126)
    set_map('num_shifts', 3)                                   # Permutations per sequence; dilation factor = multiplier × 3
    
    # Logging and output
    set_map('print_interval', 50)
    set_map('record_interval', 50)                             # Print results and save every N iterations
    
    # Decoding parameters
    set_map('decoding_threshold', 10)                          # Minimum error count per SNR point (≥1000 for OSD)
    
    # Environment settings
    set_map('Rayleigh_fading', False)                          # Must match Testing_data_gen_127 package
    set_map('reacquire_data', True)                            # True → regenerate data; False → reuse existing
```
## Execution Modes
The system operates in three distinct modes, controlled by toggling settings in `globalmap.py` and `BCH_127_testing.py`.

### Mode 1: FER Query (Performance Evaluation)
#### Configuration:
```python
# In globalmap.py
set_map('probe_MI', False)
set_map('Drawing_EXIT_only', False)
```
#### Steps:

1. Open BCH_127_testing.py in Spyder

1. Verify parameter configuration

1. Click Run File in Spyder's toolbar

**Output**: Detected QBP decoding failures are saved per SNR point in the designated output directory. These files serve as inputs for DIA model training and subsequent OSD post-processing.

### Mode 2: Data Collection for Figure Generation
#### Configuration:
```python
# In globalmap.py
set_map('probe_MI', True)
set_map('validate_dataset_size', 10)
set_map('Drawing_EXIT_only', False)
```
#### Steps:

1. Open BCH_127_testing.py in Spyder

1. Verify parameter configuration

1. Click Run File in Spyder's toolbar

**Process**: Data collection runs until `validate_dataset_size` is reached for each SNR point, then automatically stops and saves results.

### Mode 3: Figure Rendering and Saving
#### Configuration:
```python
# In globalmap.py
set_map('probe_MI', False)
set_map('validate_dataset_size', 10)
set_map('Drawing_EXIT_only', True)

# In BCH_127_testing.py - Select one key attribute
# key_attribute = 'num_shifts-'
key_attribute = 'num_rows-'       
# key_attribute = 'Others'

# In ms_test.py - Select one plotting function (uncomment as needed)
# Model.plot_density_strips_regular_file(full_compare_pattern, save_path, bandwidth=0.1)
Model.plot_mean_curves_multi_files(full_compare_pattern, save_path, true_fer_asymptotes=fer_tuple)    
# Model.plot_diff_redundancy_shift(full_compare_pattern, save_path, keystr=key_attribute)
```
**Caveat**: The selected key_attribute in BCH_127_testing.py must align with the chosen plotting function in ms_test.py to generate meaningful figures.

#### Steps:

1. Open BCH_127_testing.py in Spyder

1. Verify parameter configuration

1.  Click Run File in Spyder's toolbar

**Output**: Generated figures are saved in the designated output directory.

### Important Notes:
1. Resource Considerations: Adjust parameters (batch_size, num_batches) based on available computational resources

1. SNR Consistency: The [min_snr, max_snr] range must exactly match the settings in the Testing_data_gen_127 package

1. File Requirements: Ensure the specified parity-check matrix file exists in the project directory

1.  OSD Effectiveness: Set decoding_threshold ≥ 1000 for meaningful OSD post-processing results

## Project Structure

```
├── BCH_127_testing.py          # 🎯 Main testing script (entry point)
├── BCH_127_64_10_strip.alist   # 📊 BCH parity-check matrix
├── data_files/                 # 📂 SNR-point data files
└── figures/                    # 🖼️ Generated figures
```

## Quick Reference Card

| Action | File | Key Setting |
|--------|------|-------------|
| **Run FER tests** | `globalmap.py` | `probe_MI=False`, `Drawing_EXIT_only=False` |
| **Collect plot data** | `globalmap.py` | `probe_MI=True`, `Drawing_EXIT_only=False` |
| **Generate figures** | `globalmap.py` + `BCH_127_testing.py` + `ms_test.py` | `Drawing_EXIT_only=True` + aligned key attributes |

