# Training Guide for BCH (255,239)

## Development Environment

* **IDE**: Spyder (Anaconda Distribution)
* **OS**: Windows 10
* **Entrance File**: `BCH_255_training.py`

---

## Quick Start

### Configuration Setup

In the entrance file `BCH_255_training.py`, configure the parameters as follows:

```
sys.argv = "python 3.5 6.0 6 20 1000 10 BCH_255_239_strip.alist Check-SF1".split()  

```
for 

```
sys.argv = "python <min_snr> <max_snr> <number of SNRs> <batch_size> <num_batches> <max_iterations> <parity_check_matrix_file> <decoder_type>".split()
```
#### Parameter Description

* **min_snr**: Minimum SNR value in dB (e.g., `3.5`)
* **max_snr**: Maximum SNR value in dB (e.g., `6.0`)
* **num_SNRs**: Number of SNRs in the interval (e.g., `6`)
* **batch_size**: Number of samples per batch (e.g., `100`)
* **num_batches**: Total number of batches to process (e.g., `100`)
* **max_iterations**: Number of iterations for NMS (e.g., `10`; higher values may provide marginal FER improvement, but it depends.)
* **parity_check_matrix_file**: BCH parity-check matrix file (e.g., `BCH_255_239_strip.alist`)
* **decoder_type**: One of the BP variants (e.g., `NMS-1` refers to the Enhance NMS with a single evaluation parameter. `SPA-1` refers to the Quasi-BP (QBP) with a single evaluation parameter. `SPA-SF1(2/3)` refers to the substition variants of check nodes update equation in QBP decoder.)

---

### Execution

1. Open `BCH_255_training.py` in Spyder.
2. Ensure the configuration line matches your desired parameters.
3. Set parameters in `globalmap.py' file:
    set_map('training_model_phase',True)
    set_map('collect_failure_phase',False)
    set_map('generate_check_data',True)
4. Click the **Run File** icon in Spyder’s toolbar.

The generation of training samples for any check-node substitution (i.e. Neural Network Model) will start off by collecting all the variable-to-check and check-to-variable messages per check node in `SPA-1` decoder for the given SNRs, then the data files are merged and saved in the designated output directory.
5. If the data file for training the check-node substitution (i.e.Neural Network Model) already exists, 
Set parameter in `globalmap.py' file:
    set_map('generate_check_data',False)
6. More document of 'how to use' will be released tomorrow.

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
