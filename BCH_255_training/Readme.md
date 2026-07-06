
```markdown
# Training Guide for BCH (255,239) Decoders

This guide explains how to train and evaluate various BP-based decoders for the BCH (255,239) code.

---

## Development Environment

- **IDE**: Spyder (Anaconda Distribution)
- **OS**: Windows 10
- **Entry File**: `BCH_255_training.py`

---

## Quick Start

### 1. Configure Parameters

In `BCH_255_training.py`, set the command-line arguments as follows:

```python
sys.argv = "python 3.5 6.0 6 20 1000 10 BCH_255_239_strip.alist Check-SF1".split()
```

The command format is:

```python
sys.argv = "python <min_snr> <max_snr> <num_SNRs> <batch_size> <num_batches> <max_iterations> <parity_check_matrix_file> <decoder_type>".split()
```

#### Parameter Description

| Parameter | Description |
| :--- | :--- |
| `min_snr` | Minimum SNR value in dB (e.g., `3.5`) |
| `max_snr` | Maximum SNR value in dB (e.g., `6.0`) |
| `num_SNRs` | Number of SNR points evenly distributed between `min_snr` and `max_snr` (e.g., `6`) |
| `batch_size` | Number of samples per batch (e.g., `20`) |
| `num_batches` | Total number of batches to process (e.g., `1000`) |
| `max_iterations` | Maximum number of decoding iterations (e.g., `10`). Higher values may provide marginal FER improvement. |
| `parity_check_matrix_file` | BCH parity-check matrix file (e.g., `BCH_255_239_strip.alist`) |
| `decoder_type` | Decoder variant to use (see [Decoder Types](#decoder-types) below) |

---

### 2. Decoder Types

| Decoder Type | Description |
| :--- | :--- |
| `NMS-1` | Enhanced NMS decoder with a single trainable parameter. |
| `SPA-1` | Quasi-BP (QBP) decoder with a single trainable parameter. |
| `Check-SF1/2/3` | Check-node substitution variant for collecting NN training samples. |
| `QBP-SF1/2/3` | QBP decoder with fine-tuned NN substitution parameters. |

---

### 3. Run the Script

1. Open `BCH_255_training.py` in Spyder.
2. Verify that the configuration line matches your desired parameters.
3. Click the **Run File** icon (green triangle) in Spyder's toolbar.

---

## Workflow Overview

The overall workflow depends on the chosen `decoder_type`. All mode switches are controlled via `globalmap.py`.

### Phase Control (in `globalmap.py`)

| Flag | Description |
| :--- | :--- |
| `training_model_phase` | `True` = training mode, `False` = evaluation/collection mode |
| `collect_failure_phase` | `True` = collect decoding failures for DIA training, `False` = skip |
| `generate_check_data` | `True` = generate NN training samples for check-node substitution, `False` = use existing samples |

---

### Case 1: `NMS-1` or `SPA-1`

These decoders have a **single trainable parameter**. You can either:

- **Skip training**: Directly evaluate FER and collect failed samples using a pre-determined parameter (e.g., from rough line search).
- **Train first**: Optimize the parameter, then evaluate FER and collect failures.

**Switch between phases** in `globalmap.py`:

```python
# Training phase
set_map('training_model_phase', True)
set_map('collect_failure_phase', False)

# Evaluation/collection phase
set_map('training_model_phase', False)
set_map('collect_failure_phase', True)
```

---

### Case 2: `QBP-SF1/2/3` (with NN Check-Node Substitution)

This workflow involves **four stages**:

#### Stage 1: Collect Training Samples for NN Substitution

- Set `decoder_type = "Check-SF1"`, `"Check-SF2"`, or `"Check-SF3"` temporarily.
- In `globalmap.py`, set:
  ```python
  set_map('training_model_phase', True)
  set_map('collect_failure_phase', False)
  set_map('generate_check_data', True)
  ```
- Run the script. All variable-to-check and check-to-variable messages from the `SPA-1` decoder will be collected, merged, and saved as a single training samples file.
- **If the samples file already exists**, set `generate_check_data = False` to save time.

#### Stage 2: Train the NN Substitution

- Ensure the training samples file is available, then configure training settings in `globalmap.py`:

```python
    set_map('initial_learning_rate', 0.001)
    set_map('decay_rate', 0.99)
    set_map('decay_step', 200)
    set_map('iterate_termination_step', 100000)  # Training stops after this many steps
```
- Run the script to train the NN substitution model.

#### Stage 3: Fine-Tune `QBP-SF1/SF2/SF3`

- Set `decoder_type = "QBP-SF1"`, `"QBP-SF2"`, or `"QBP-SF3"`.
- In `globalmap.py`, set:
  ```python
  set_map('training_model_phase', True)
  set_map('collect_failure_phase', False)
  set_map('generate_check_data', False) 
  ```
- Run the script to fine-tune the combined parameters **separately for each SNR point**, yielding a dedicated model per SNR value.

#### Stage 4: Evaluate and Collect Failures
- In `globalmap.py`, set:
 ```python
  set_map('training_model_phase', False)
  set_map('collect_failure_phase', True)
 ```
- Run the script to evaluate FER and collect failed samples (threshold: `set_map('termination_threshold', 100)`).

## Failure Trajectories for DIA Training

The decoding failure trajectories from **all decoder types** (`NMS-1`, `SPA-1`, `QBP-SF1/2/3`) are automatically stored. These trajectories serve as training samples for the **DIA model**, which enhances the performance of Ordered Statistics Decoding (OSD) post-processing.

---

## Global Settings (`globalmap.py`)

### Training Hyperparameters

```python
set_map('initial_learning_rate', 0.001)
set_map('decay_rate', 0.99)
set_map('decay_step', 200)
set_map('iterate_termination_step', 100)   # Adam optimizer termination step
```

### Decoding Configuration

```python
set_map('reduction_iteration', 4)          # Iterations for parity-check matrix row reduction
set_map('redundancy_factor', 2)            # Redundancy factor for matrix row count
set_map('num_shifts', 3)                   # Allowed shifts per received sequence
```

### Output & Logging

```python
set_map('print_interval', 20)
set_map('record_interval', 20)             # Print and save model every N intervals
```

### Matrix Options

```python
set_map('regular_matrix', False)                    # Disable conventional parity-check matrix
set_map('generate_extended_parity_check_matrix', True)  # Enable optimized matrix with redundant rows
```

### Model Checkpoint

```python
def logistic_setting():
    restore_step = 'latest'   # '' starts fresh; 'latest' loads the most recent model
```

---

## Important Notes

- Adjust parameters according to your available computational resources and training requirements.
- The SNR range `[min_snr, max_snr]` must **strictly match** the corresponding settings in the `Training_data_gen_255` package.
- Ensure the parity-check matrix file is located in the project directory.

---

## Project Structure

```
.
├── BCH_255_training.py              # 🎯 Main entry script
├── BCH_255_239_strip.alist          # 📊 BCH parity-check matrix
├── globalmap.py                     # ⚙️ Global configuration and mode switches
├── ckpts/                           # 🧠 Well-trained decoder models (auto-created)
│   └── <min_snr>-<max_snr>dB/       # Named after your SNR range (e.g., 3.5-6.0dB)
│       └── <decoder_type>/          # e.g., Check-SF1/2/3, NMS-1, etc.
└── data/                            # 📁 Output directory (generated by Training_data_gen_255)
    ├── check_node_training_samples/ # NN substitution training data shared by Check-SF1/2/3
    └── failure_trajectories/        # Decoding failure trajectories (for DIA training)
```

---

## License

Part of the presented code has applied for a Chinese patent and is in progress.
```
