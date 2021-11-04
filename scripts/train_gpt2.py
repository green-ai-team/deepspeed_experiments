from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from torch.optim import Adam
from datasets import ClassLabel, load_dataset
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import deepspeed
import random
import argparse
import pandas as pd
from IPython.display import display, HTML
from tqdm import tqdm
import wandb
import sys
import os


class CustomTextDataset(Dataset):
    def __init__(self, input_ids, labels):
        self.labels = labels
        self.input_ids = input_ids
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        label = self.labels[idx]
        input_id = self.input_ids[idx]
        sample = {"input_ids": input_id, "labels": label, "attention_mask": torch.ones_like(input_id)}
        return sample


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default = 20, type = int, help = 'Number of epochs')
parser.add_argument('--batch_size', default = 4, type = int, help = 'Batch size')
parser.add_argument('--learning_rate', default = 3e-4, type = float, help = 'Learning rate')
parser.add_argument('--zero_stage', default = 0, type = int, help = 'Chooses different stages of ZeRO Optimizer. Stage 0, 1, 2, and 3 refer to disabled, optimizer state partitioning, and optimizer+gradient state partitioning, and optimizer+gradient+parameter partitioning, respectively.')
parser.add_argument('--wandb_name', default='gpt2',
                    help='Name W&B will use when saving results.\ne.g. `--wandb_name "coco2017-full-sparse"`')
parser.add_argument('--wandb_entity', default=None,
                    help='Name of W&B team/entity to log to.')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
# parser = deepspeed.add_config_arguments(parser)

args = parser.parse_args()
rank = int(os.getenv('LOCAL_RANK'))
world_size = int(os.getenv('WORLD_SIZE'))
print(f'rank from deepspeed : {args.local_rank}, rank from environ: {rank}, worldsize environ: {world_size}\n', flush=True)
init_method = 'file://' + os.getenv('COMM_PATH')
print(f'Init Method: {init_method}\n', flush=True)
torch.distributed.init_process_group(
    backend='nccl',
    world_size=world_size, 
    rank=rank,
    init_method=init_method)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print('Loading Dataset', flush=True)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
block_size = tokenizer.model_max_length
with open('../data/dracula.txt') as f:
    single_string = f.read()
string_tokenized = tokenizer(single_string, return_tensors='pt')
# print('tensor:', string_tokenized['input_ids'][0], flush=True)
examples = []
for i in range(0, len(string_tokenized['input_ids'][0]) - block_size, block_size+1):
    examples.append(string_tokenized['input_ids'][0][i:i + block_size+1])
inputs, labels = [], []
for ex in examples:
    inputs.append(ex[:-1])
    labels.append(ex[1:])
inputs_labels_df = pd.DataFrame({'input_ids': inputs, 'labels': labels})
dataset = CustomTextDataset(inputs_labels_df['input_ids'], inputs_labels_df['labels'])


print('Loading model...', flush=True)
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)

print('Initializing DeepSpeed...', flush=True)
deepspeed_config = {
    'train_batch_size': args.batch_size,
#     'gradient_accumulation_steps': args.ga_steps,
#     'gradient_clipping': GRAD_CLIP_NORM,
#     'fp16': {
#         'enabled': args.fp16,
#     },
#     'amp': {
#         'enabled': args.amp,
#         'opt_level': 'O1',
#     },
#     "flops_profiler": {
#         "enabled": args.flops_profiler,
#         "profile_step": 200,
#         "module_depth": -1,
#         "top_modules": 1,
#         "detailed": True,
#         "output_file": None # TODO Can't get this to work.
#     },
    "zero_optimization": {
       "stage": args.zero_stage,
       "grad_hooks": False,
       # Offload the model parameters If you have an nvme drive - you should use the nvme option.
       # Otherwise, use 'cpu' and remove the `nvme_path` line
       # "offload_param": {
       #     "device": "cpu",
       #     # "nvme_path": "/path/to/nvme/folder",
       # },
        # Offload the optimizer of choice. If you have an nvme drive - you should use the nvme option.
        # Otherwise, use 'cpu' and remove the `nvme_path` line
#         "offload_optimizer": {
#             "device": "cpu",
            # "nvme_path": "/path/to/nvme/folder",
#        },
    },
    "optimizer": {
        "type": "Adam",  # You can also use AdamW here
        "params": {
            "lr": args.learning_rate,
        },
    },
}
model_engine, optimizer, data_loader, _ = deepspeed.initialize(args=args,
                                                               model=model,
                                                               model_parameters=model.parameters(),
                                                               training_data=dataset,
                                                               config=deepspeed_config)


print('Initializing wandb...')
if torch.distributed.get_rank() == 0 and args.wandb_entity is not None:
    run = wandb.init(
            project='gpt2',
            entity='darayavaus',
            resume=False,
          )

print('Training...', flush=True)
for epoch in range(args.epochs):
    for i, batch in enumerate(data_loader):
        batch = {
            'attention_mask': batch['attention_mask'].to(device),
            'input_ids': batch['input_ids'].to(device),
            'labels': batch['labels'].to(device)
        }
        if i == 0: 
            print(f"batch shape in rank {torch.distributed.get_rank()} = {batch['attention_mask'].shape}", flush=True)
        
        loss = model_engine(**batch).loss
        model_engine.backward(loss)
        model_engine.step()
#         print(f'did step in {torch.distributed.get_rank()}', flush=True)
            
        if i % 1 == 0:
            avg_loss = loss.detach().clone()
#             # try just reduce?
#             torch.distributed.all_reduce(avg_loss, torch.distributed.ReduceOp.SUM)
            torch.distributed.reduce(avg_loss, 0)
            avg_loss = avg_loss / world_size
            
            if torch.distributed.get_rank() == 0:
                log = {
                    'epoch': epoch,
                    'iter': i,
                    'loss': avg_loss.item()
                }
                if i % 10 == 0:
                    with torch.no_grad():
                        output = model.generate(**tokenizer("Dear Mr. ", return_tensors='pt').to(device), max_length=30)
                        columns = ["Context", "Generation"]
                        data = [["Dear Mr. ", tokenizer.decode(output[0], skip_special_tokens=True)]]
                        log['text'] = wandb.Table(data=data, columns=columns)
                if args.wandb_entity is not None:
                    wandb.log(log)
                else:
                    print(f"epoch: {log['epoch']}, iter: {log['iter']}, loss: {log['loss']}", flush=True)
#         if i % 10 == 0 and torch.distributed.get_rank() == 0:
#             print(f'epoch:{epoch}, iter: {i}', flush=True)
            
            
            
if torch.distributed.get_rank() == 0:
    wandb.finish()

print('Done!', flush=True)