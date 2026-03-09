# BCH (127,64) Testing Data Generation Guide

## Development Environment

* **IDE**: Spyder (Anaconda Distribution)
* **OS**: Windows 10
* **Entrance File**: `Main_test_gen.py`

---

## Quick Start

### Configuration Setup

In the entrance file `Main_test_gen.py`, configure the parameters as follows:

```
sys.argv = "python 2.0 4.5 6 100 100 BCH_127_64_10_strip.alist".split()
```
for 

```
sys.argv = "python <min_snr> <max_snr> <num_points> <batch_size> <max_num_batches> <parity_check_matrix_file>".split()
```

#### Parameter Description

* **min_snr**: Minimum SNR value in dB (e.g., `2.0`)
* **max_snr**: Maximum SNR value in dB (e.g., `4.5`)
* **num_points**: Number of SNR points evenly distributed between min_snr and max_snr (e.g., `6`)
* **batch_size**: Number of samples per batch (e.g., `100`)
* **max_num_batches**: Total number of batches to generate (e.g., `100`)
* **parity_check_matrix_file**: BCH parity-check matrix file (e.g., `BCH_127_64_10_strip.alist`)

---

### Execution

1. Open `Main_test_gen.py` in Spyder.
2. Ensure the configuration line matches your desired parameters.
3. Click the **Run File** icon in Spyder's toolbar.

Training data will be generated and stored in the designated output directory.

---
## Settings Overview (in `Main_test_gen.py`)

```python
GL.set_map('Rayleigh_fading', False)   #AWGN data by default, otherwise data for the 'True' evaluation  for 'Raleigh_fading''
```

---

## Notes

* Adjust parameters according to your computational resources and testing needs.
* The SNR range `[min_snr, max_snr]` creates evenly distributed batches across the specified interval with designated proportions to the maximum 
number of batches given in **max_num_batches**.
* Ensure the parity-check matrix file is accessible within the project directory.

---

## Project Structure

```
├── Main_test_gen.py          # 🎯 Main testing data generating script (Entrance file)
├── BCH_127_64_10_strip.alist  # 📊 BCH parity-check matrix
└── [Generated testing data list→ output directories]
```
