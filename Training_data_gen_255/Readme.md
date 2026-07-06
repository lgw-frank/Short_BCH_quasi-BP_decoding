# BCH (255,239) Training Data Generation Guide

This guide explains how to generate training data for the BCH (255,239) code using the provided script.

---

## Development Environment

- **IDE**: Spyder (Anaconda Distribution)
- **OS**: Windows 10
- **Entry File**: `Main_train_gen.py`

---

## Quick Start

### 1. Configure Parameters

Open `Main_train_gen.py` and modify the configuration line (see inline comments for other settings) as follows:

```python
sys.argv = "python 3.5 6.0 6 100 1000 BCH_255_239_strip.alist".split()
```

The command format is:

```python
sys.argv = "python <min_snr> <max_snr> <num_points> <batch_size> <max_num_batches> <parity_check_matrix_file>".split()
```

#### Parameter Description

| Parameter | Description |
| :--- | :--- |
| `min_snr` | Minimum SNR value in dB (e.g., `3.5`) |
| `max_snr` | Maximum SNR value in dB (e.g., `6.0`) |
| `num_points` | Number of SNR points evenly distributed between `min_snr` and `max_snr` (e.g., `6`) |
| `batch_size` | Number of samples per batch (e.g., `100`) |
| `max_num_batches` | Total number of batches to generate (e.g., `1000`) |
| `parity_check_matrix_file` | BCH parity-check matrix file (e.g., `BCH_255_239_strip.alist`) |

---

### 2. Run the Script

1. Open `Main_train_gen.py` in Spyder.
2. Verify that the configuration line matches your desired parameters.
3. Click the **Run File** icon (green triangle) in Spyder's toolbar.

Training data for each SNR point will be generated and saved in the designated output directories.



## Important Notes

- Adjust parameters according to your available computational resources and training requirements.
- The SNR range `[min_snr, max_snr]` is divided evenly into `num_points` intervals. Each SNR point generates `max_num_batches` batches of size `batch_size`.
- Ensure `BCH_255_239_strip.alist` (or your specified parity-check matrix file) is located in the **same directory** as `Main_train_gen.py`.

---

## Project Structure

```
.
├── Main_train_gen.py # 🎯 Main script (entry point)
├── BCH_255_239_strip.alist # 📊 BCH parity-check matrix
└── data/ # 📁 Generated training data
└── snr<min_snr>-<max_snr>dB/ # 📂 Named after your SNR range (e.g., 3.5-6.0dB)
├── <SNR_1>dB/ # Data for first SNR point
├── <SNR_2>dB/ # Data for second SNR point
└── ... # One subfolder per SNR point
```

---

## Troubleshooting

- **File not found error**: Make sure the `.alist` file is in the same folder as the script.
- **Memory issues**: Reduce `batch_size` or `max_num_batches` if you encounter out-of-memory errors.
- **SNR range**: Ensure `min_snr < max_snr` and `num_points >= 2` for a meaningful distribution.

---

## License

 MIT