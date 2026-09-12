# Global Neural Adjoint for Optical Multilayer Thin-Film Inverse Design

This repository provides a cleaned and modularized implementation of the OMT-mixer and Global Neural Adjoint (GNA) framework used in the manuscript:

**“Interpretable inverse design of optical multilayer thin films based on global neural adjoint.”**

The code includes the OMT-mixer forward surrogate, GNA-based inverse design, material and thickness loss functions, constraint handling, sensitivity-profile calculation, and example data for verification.

## Repository structure

```text
GNA/
├── README.md
├── requirements.txt
├── checkpoints/
├── configs/
├── example_data/
├── refractive_index_data/
├── scripts/
├── src/
│   └── gna_omt/
└── tests/

Installation

Install the required packages using:

pip install -r requirements.txt

The code automatically uses CUDA when a compatible GPU is available and otherwise falls back to CPU.

Pretrained OMT-mixer

Place the pretrained OMT-mixer checkpoint in:

checkpoints/OMT_Mixer.pth

The supplied checkpoint corresponds to the OMT-mixer used for the inverse-design experiments reported in the manuscript.

Model verification

To verify the pretrained model, run:

python scripts/check_model.py

The script evaluates the six example OMT structures provided in example_data/.

For the supplied checkpoint, the mean RMSE over the six example structures is approximately 0.005.

Running GNA

Run the GNA inverse-design example using:

python scripts/run_gna_main.py

The default example uses Test1 as the target spectrum.

For a quick functional check, a reduced population size and number of iterations may be used. The main experimental settings used in the manuscript are:

Population size: 3,000
Optimization iterations: 1,000
Admissible layer numbers: 4, 8, 12, 16, 20, and 24
Thickness range: 20–100 nm
Number of candidate materials: 11

Detailed hyperparameters are provided in the manuscript and configuration files.

Citation

If you use this code in your research, please cite:

S. Kim and J. Kim,
“Interpretable inverse design of optical multilayer thin films based on global neural adjoint,”
Scientific Reports, publication information to be updated.