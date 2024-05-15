![LOGO](https://github.com/DeepWave-KAUST/latentpinnn-dev/blob/main/asset/latentpinn.png)

:vertical_traffic_light: A copy of this repository has been made public at [LINK](https://github.com/DeepWave-KAUST/latentpinn). All future developments will be initially made available to sponsors in this repository. :vertical_traffic_light:

Reproducible material for **DW0037 - Mohammad H. Taufik, Xinquan Huang, Tariq Alkhalifah.**

[Click here](https://kaust.sharepoint.com/:f:/r/sites/M365_Deepwave_Documents/Shared%20Documents/Restricted%20Area/DW0037?csf=1&web=1&e=fHFEeB) to access the Project Report. Authentication to the _Restricted Area_ filespace is required.

# Project structure
This repository is organized as follows:

* :open_file_folder: **asset**: folder containing logo.
* :open_file_folder: **data**: a folder containing the subsampled velocity models used to train the PINN.
* :open_file_folder: **notebooks**: reproducible notebook for the synthetic tests of the paper.
* :open_file_folder: **scripts**: script examples to perform autoencoder training, PINNs training to solve for the eikonal and scattered Helmholtz equations.
* :open_file_folder: **saves**: a folder containing the trained PINN model.
* :open_file_folder: **src**: a folder containing routines for the `latentpinn` source file.

## Getting started :space_invader: :robot:
To ensure the reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

To install the environment, run the following command:
```
./install_env.sh
```
It will take some time, but if, in the end, you see the word `Done!` on your terminal, you are ready to go. 

Remember to always activate the environment by typing:
```
conda activate latentpinn
```

**Disclaimer:** All experiments have been carried on a Intel(R) Xeon(R) Silver 4316 CPU @ 2.30GHz equipped with a single NVIDIA A100 GPU. Different environment 
configurations may be required for different combinations of workstation and GPU.

## Cite us 
```bibtex
@article{taufik2023latentpinns,
  title={LatentPINNs: Generative physics-informed neural networks via a latent representation learning},
  author={Taufik, Mohammad H and Alkhalifah, Tariq},
  journal={arXiv preprint arXiv:2305.07671},
  year={2023}
}
@article{taufik2024multiple,
  title={Multiple Wavefield Solutions in Physics-Informed Neural Networks using Latent Representation},
  author={Taufik, Mohammad H and Huang, Xinquan and Alkhalifah, Tariq},
  journal={IEEE Geoscience and Remote Sensing Letters},
  year={2024},
  publisher={IEEE}
}
