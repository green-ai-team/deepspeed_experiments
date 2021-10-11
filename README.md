# deepspeed_experiments

## Environment setup for deepspeed zero-infinity usage (for zhores):
- ```module load python/anaconda3 compilers/cmake-3.20 compilers/gcc-8.3.0 gpu/cuda-11.1```
- ```conda create --name ENV python=3.8```
- ```conda activate ENV```
- ```conda install -c conda-forge python-libaio```
- ```conda install -c conda-forge llvm-tools```
- ```CUDA_PATH=/trinity/shared/opt/cuda-11.1```
- ```CUDA_HOME=/trinity/shared/opt/cuda-11.1```
- ```pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f [https://download.pytorch.org/whl/torch_stable.html](https://download.pytorch.org/whl/torch_stable.html)```
- ```pip3 install deepspeed```
- ```pip3 install dalle-pytorch```
- ```pip3 install requests pyyaml tqdm packaging transformers psutil wandb```

Zero-offload example in ```notebooks/Dalle-Zero.ipynb```
