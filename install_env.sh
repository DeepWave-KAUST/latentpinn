#!/bin/bash
# 
# Installer for package
# 
# Run: ./install_env.sh

echo 'Creating Package environment'

# Create conda env
# source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || 
source ~/miniconda3/etc/profile.d/conda.sh
conda env create -f environment.yml
conda activate latentpinn

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip3 install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip3 install -e git+https://github.com/openai/CLIP.git@main#egg=clip
pip3 install -e git+https://github.com/CompVis/latent-diffusion.git@main#egg=latent-diffusion

# Install latentpinn package
pip3 install -e . --use-pep517

conda env list
echo 'Created and activated environment:' $(which python)

# Check cupy works as expected
echo 'Checking cupy version and running a command...'
python -c 'import torch; print(torch.__version__);  print(torch.cuda.get_device_name(torch.cuda.current_device())); print(torch.ones(10).to("cuda:0"))'

echo 'Done!'

