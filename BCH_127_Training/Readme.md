# Training Guide for BCH (127,64)

## Development Environment

* **IDE**: Spyder (Anaconda Distribution)
* **OS**: Windows 10
* **Entrance File**: `BCH_127_training.py`

---

## Quick Start

### Configuration Setup

In the entrance file `BCH_127_training.py`, configure the parameters as follows:

```
sys.argv = "python 3.0 3.0 100 100 8 BCH_127_64_10_strip.alist NMS-1".split()
```
for 

```
sys.argv = "python <min_snr> <max_snr> <batch_size> <num_batches> <max_iterations> <parity_check_matrix_file> <decoder_type>".split()
```
#### Parameter Description

* **min_snr**: Minimum SNR value in dB (e.g., `3.0`)
* **max_snr**: Maximum SNR value in dB (e.g., `3.0`)
* **batch_size**: Number of samples per batch (e.g., `100`)
* **num_batches**: Total number of batches to process (e.g., `100`)
* **max_iterations**: Number of iterations for NMS (e.g., `8`; higher values provide only marginal FER improvement; `4` is recommended for code lengths below `100`, and `8` for lengths above `100`.)
* **parity_check_matrix_file**: BCH parity-check matrix file (e.g., `BCH_127_64_10_strip.alist`)
* **decoder_type**: One of the BP variants (e.g., `NMS-1` refers to the conventional NMS with a single evaluation parameter.)

---

### Execution

1. Open `BCH_127_training.py` in Spyder.
2. Ensure the configuration line matches your desired parameters.
3. Click the **Run File** icon in Spyder’s toolbar.

Training will start, and results will be periodically saved in the designated output directory.
After training completes, the model undergoes final parameter evaluation using the validation dataset.
Additionally, the trajectories of NMS decoding failures are stored for use as training samples in the DIA model.

---

## Settings Overview (in `globalmap.py`)

```python
def global_setting(argv):

    set_map('initial_learning_rate', 0.01)
    set_map('decay_rate', 0.95)
    set_map('decay_step', 500)
    set_map('nms_termination_step', 200) # Adam optimizer terminates after 'nms_termination_step' steps.     
    
    set_map('reduction_iteration',4)     #number of iterations used to acquire parity-check matrix rows with minimal weights   
    set_map('redundancy_factor',2)       # redundancy factor used to regulate the number of rows in the parity-check matrix
    set_map('num_shifts',3)              # number of shifts allowed per received sequence
    
    set_map('print_interval',20)
    set_map('record_interval',20)       # Print results and save model every interval
    
    set_map('regular_matrix',False)     #disable use of the conventional parity-check matrix 
    set_map('generate_extended_parity_check_matrix',True)  #enable optimized parity-check matrix with redundant rows for enhanced NMS decoding
    
    set_map('enhanced_NMS_indicator',True)  #enable enhanced NMS decoder
    set_map('original_NMS_indicator',False) #disable conventional NMS decoder; both switches must align with the chosen parity-check matrix. 
def logistic_setting():
    restore_step = 'latest' # '' starts fresh; 'latest' loads the most recent model.
```

---

## Notes

* Adjust parameters according to your computational resources and training requirements.
* The SNR range `[min_snr, max_snr]` must strictly match the corresponding settings in the `Training_data_gen_127` package
* Ensure the parity-check matrix file is accessible within the project directory.

---

## Project Structure

```
├── BCH_127_training.py       # 🎯 Main training script (Entrance file)
├── BCH_127_64_10_strip.alist  # 📊 BCH parity-check matrix
├── [Well-trained NMS decoder → output directory]
└── [data file containing NMS decoding failure trajectories used as DIA training samples → output directory]
```
