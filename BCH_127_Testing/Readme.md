# BCH (127,64) Testing Guide

## Development Environment

* **IDE**: Spyder (Anaconda Distribution)
* **OS**: Windows 10
* **Entrance File**: `BCH_127_testing.py`

---

## Quick Start

### Configuration Setup

In the entrance file `BCH_127_testing.py`, configure the parameters as follows:

```python
sys.argv = "python 2.0 4.5 6 100 1000 8 BCH_127_64_10_strip.alist NMS-1".split()
```
for 

```
sys.argv = "python <min_snr> <max_snr> <num_points> <batch_size> <num_batches> <max_iterations> <parity_check_matrix_file> <decoder_type>".split()
```
#### Parameter Description

* **min_snr**: Minimum SNR value in dB (e.g., `2.0`)
* **max_snr**: Maximum SNR value in dB (e.g., `4.5`)
* **num_points**: Number of SNR points evenly distributed between min_snr and max_snr (e.g., `6`)
* **batch_size**: Number of samples per batch (e.g., `100`)
* **num_batches**: Total number of batches to process (e.g., `100`)
* **max_iterations**: Number of iterations for NMS (e.g., `8`; larger values yield limited FER gain)
* **parity_check_matrix_file**: BCH parity-check matrix file (e.g., `BCH_127_64_10_strip.alist`)
* **decoder_type**: One of the BP variants (e.g., `NMS-1` refers to the conventional NMS with a single parameter to evaluate)

---

### Execution

1. Open `BCH_127_testing.py` in Spyder.
2. Ensure the configuration line matches your desired parameters.
3. Click the **Run File** icon in Spyder's toolbar.

Testing will begin, and detected NMS decoding failures will be saved in the designated output directory for each SNR point.
These saved data files will later serve as testing inputs for the DIA model and subsequently feed into OSD post-processing.

---

## Settings Overview (in `globalmap.py`)

```python
def global_setting(argv):

    set_map('regular_matrix',False)     #disable use of the conventional parity-check matrix 
    set_map('generate_extended_parity_check_matrix',True)  #enable optimized parity-check matrix with redundant rows for enhanced NMS decoding 
    set_map('reduction_iteration',4)     #number of iterations used to acquire parity-check matrix rows with minimal weights   
    set_map('redundancy_factor',2)       # redundancy factor used to regulate the number of rows in the parity-check matrix
    set_map('num_shifts',3)              # number of shifts allowed per received sequence
    
    set_map('print_interval',50)
    set_map('record_interval',50)       # Print results and save model every interval
    
    set_map('decoding_threshold',10)    # the minimum errors collected for each SNR point. Should be set high enough (≥1000) for OSD post-processing to be effective..
    set_map('Rayleigh_fading', False)   # Must match the fading parameter setting in `Testing_data_gen_127` package
    set_map('reacquire_data',True)      # True → regenerate NMS failure data files; False → reuse existing files

def logistic_setting():
    restore_step = 'latest'             # Required to load the most recent model
```

---

## Notes

* Adjust parameters according to your computational resources and testing needs.
* The SNR range `[min_snr, max_snr]` must strictly match the settings used in the `Testing_data_gen_127` package
* Ensure the specified parity-check matrix file is present in the project directory.

---

## Project Structure

```
├── BCH_127_testing.py        # 🎯 Main testing script (Entrance file)
├── BCH_127_64_10_strip.alist  # 📊 BCH parity-check matrix
├── decoding performance       # 📂 Directory storing results and logs
└── failures                   # 📂 Saved NMS decoding failures per SNR point
└── plots                      # 📂 Plots of iteration distributions across SNR points
```
