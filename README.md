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
- ```pip install deepspeed```
- ```pip install dalle-pytorch```
- ```pip install requests pyyaml tqdm packaging transformers psutil wandb```

Check torch and torchvision versions with ```pip freeze``` and ```conda list```. There might be different versions installed by both pip and conda. Leave only the 1.8.1+cu111 version.

Example output of ```ds_report``` command:
```
--------------------------------------------------
DeepSpeed C++/CUDA extension op report
--------------------------------------------------
NOTE: Ops not installed will be just-in-time (JIT) compiled at
      runtime if needed. Op compatibility means that your system
      meet the required dependencies to JIT install the op.
--------------------------------------------------
JIT compiled ops requires ninja
ninja .................. [OKAY]
--------------------------------------------------
op name ................ installed .. compatible
--------------------------------------------------
cpu_adam ............... [NO] ....... [OKAY]
fused_adam ............. [NO] ....... [OKAY]
fused_lamb ............. [NO] ....... [OKAY]
sparse_attn ............ [NO] ....... [OKAY]
transformer ............ [NO] ....... [OKAY]
stochastic_transformer . [NO] ....... [OKAY]
async_io ............... [NO] ....... [OKAY]
transformer_inference .. [NO] ....... [OKAY]
utils .................. [NO] ....... [OKAY]
quantizer .............. [NO] ....... [OKAY]
--------------------------------------------------
No CUDA runtime is found, using CUDA_HOME='/trinity/shared/opt/cuda-11.1'
DeepSpeed general environment info:
torch install path ............... ['/home/USERNAME/.conda/ENV/lib/python3.8/site-packages/torch']
torch version .................... 1.8.1+cu111
torch cuda version ............... 11.1
nvcc version ..................... 11.1
deepspeed install path ........... ['/home/USERNAME/.conda/ENV/lib/python3.8/site-packages/deepspeed']
deepspeed info ................... 0.5.4, unknown, unknown
deepspeed wheel compiled w. ...... torch 1.9, cuda 10.2
```

Zero-offload example in ```notebooks/Dalle-Zero.ipynb```
