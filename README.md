# Short_BCH_quasi-BP_decoding

## Project Overview

This project implements quasi-BP (Belief Propagation) decoding algorithms for short BCH codes. The codebase is organized into two main submodules:

| Module | Purpose | Status |
| :--- | :--- | :--- |
| [`BCH_255_Training/`](./BCH_255_Training/) | **Main module** for training, decoding, and evaluation | 🟢 Active development |
| [`Training_data_gen_255/`](./Training_data_gen_255/) | Generates and saves raw received sequences for decoding | 🟡 Maintenance mode |

---

## ⚠️ Strong Recommendations

**Before working with any module, please read its corresponding `Readme.md` file.**

Each module has its own dependencies, configuration requirements, and usage workflows. The module-specific `Readme.md` provides:
- Setup instructions and environment configuration
- Parameter descriptions and usage examples
- Workflow guidance for training, evaluation, and data generation

Starting with the module's documentation will save you time and help avoid common issues.

---

## Quick Start

1. **Clone the repository**:
   ```bash
   git clone git@github.com:lgw-frank/Short_BCH_quasi-BP_decoding.git
   cd Short_BCH_quasi-BP_decoding
   ```

2. **Choose your workflow**:
   - For **training decoders**: start with [`BCH_255_Training/`](./BCH_255_Training/)
   - For **generating training data**: start with [`Training_data_gen_255/`](./Training_data_gen_255/)

3. **Read the module-specific `Readme.md`** before running any scripts.

---

## Repository Structure

```
.
├── BCH_255_Training/          # 🎯 Main module: training and evaluation
│   └── Readme.md              # Detailed guide for decoder training
├── Training_data_gen_255/     # 📊 Supporting module: data generation
│   └── Readme.md              # Detailed guide for data generation
└── Readme.md                  # This file
```

---

## License
This project is released under the MIT License. Please be aware that specific portions of the code are covered by a pending Chinese patent. For clarification on which parts are affected, please contact the author.
```
