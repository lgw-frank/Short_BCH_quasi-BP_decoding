# BCH (255,239) Training Data Generation Guide

## Development Environment

* **IDE**: Spyder (Anaconda Distribution)
* **OS**: Windows 10
* **Entrance File**: `Main_train_gen.py`

---

## Quick Start

### Configuration Setup

In the entrance file `Main_train_gen.py`, configure the parameters as follows:

```
sys.argv = "python 3.5 6.0 6 100 1000 BCH_255_239_strip.alist".split()
 
```
for 

```
sys.argv = "python <min_snr> <max_snr> <num_points> <batch_size> <max_num_batches> <parity_check_matrix_file>".split()
```

#### Parameter Description

* **min_snr**: Minimum SNR value in dB (e.g., `3.5`)
* **max_snr**: Maximum SNR value in dB (e.g., `6.0`)
* **num_points**: Number of SNR points evenly distributed between min_snr and max_snr (e.g., `6`)
* **batch_size**: Number of samples per batch (e.g., `100`)
* **max_num_batches**: Total number of batches to generate (e.g., `1000`)
* **parity_check_matrix_file**: BCH parity-check matrix file (e.g., `BCH_255_239_strip.alist`)

---

### Execution

1. Open `Main_train_gen.py` in Spyder.
2. Ensure the configuration line matches your desired parameters.
3. Click the **Run File** icon in Spyder's toolbar.

Training data will be generated and stored in the designated output directory.

---
## Settings Overview (in `Main_train_gen.py`)
---

## Notes

* Adjust parameters according to your computational resources and testing needs.
* The SNR range `[min_snr, max_snr]` creates evenly distributed batches across the specified interval with designated proportions to the maximum 
number of batches given in **max_num_batches**.
* Ensure the parity-check matrix file is accessible within the project directory.

---

## Project Structure

```
├── Main_train_gen.py          # 🎯 Main training data generating script (Entrance file)
├── BCH_255_239_strip.alist  # 📊 BCH parity-check matrix
└── [Generated training data list→ output directories]
```
