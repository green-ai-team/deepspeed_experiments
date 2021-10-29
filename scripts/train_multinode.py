import argparse
from pathlib import Path
import time
from glob import glob
import os
import shutil

import torch
import torch.profiler
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp


def main():

    print('Python code launched on node', os.environ['SLURM_PROCID'], flush=True)
    # argument parsing

    parser = argparse.ArgumentParser()

    
    parser.add_argument('--bpe_path', type=str,
                        help='path to your BPE json file')

    parser.add_argument('--dalle_output_file_name', type=str, default = "dalle",
                        help='output_file_name')

    parser.add_argument('--fp16', action='store_true',
                        help='(experimental) - Enable DeepSpeed 16 bit precision. Reduces VRAM.')

    parser.add_argument('--amp', action='store_true',
        help='Apex "O1" automatic mixed precision. More stable than 16 bit precision. Can\'t be used in conjunction with deepspeed zero stages 1-3.')

    parser = distributed_utils.wrap_arg_parser(parser)

    train_group = parser.add_argument_group('Training settings')

    train_group.add_argument('--epochs', default = 20, type = int, help = 'Number of epochs')

    train_group.add_argument('--save_every_n_steps', default = 1000, type = int, help = 'Save a checkpoint every n steps')

    train_group.add_argument('--batch_size', default = 4, type = int, help = 'Batch size')

    train_group.add_argument('--ga_steps', default = 1, type = int, help = 'Number of steps to accumulate gradients across per each iteration. DeepSpeed only.')
    
    parser.add_argument('--zero_stage', default = 0, type = int, help='ZeRO stages')
    args = parser.parse_args()
    
    mp.spawn(train, nprocs=1, args=(args,))

# helpers

def exists(val):
    return val is not None

def get_trainable_params(model):
    return [params for params in model.parameters() if params.requires_grad]

def cp_path_to_dir(cp_path, tag):
    """Convert a checkpoint path to a directory with `tag` inserted.
    If `cp_path` is already a directory, return it unchanged.
    """
    if not isinstance(cp_path, Path):
        cp_path = Path(cp_path)
    if cp_path.is_dir():
        return cp_path
    path_sans_extension = cp_path.parent / cp_path.stem
    cp_dir = Path(f'{path_sans_extension}-{tag}-cp')
    return cp_dir


def train(gpu, args):

    # initialize distributed backend
    rank = int(os.environ['SLURM_PROCID'])

    distr_backend = distributed_utils.set_backend_from_args(args)
    print('distributed_backend', distr_backend, flush=True)
    distr_backend.initialize()

    using_deepspeed = True

    print('Using DeepSpeed:', using_deepspeed, flush=True)

    # distribute
    distr_backend.check_batch_size(BATCH_SIZE)
    deepspeed_config = {
        'train_batch_size': BATCH_SIZE,
        'gradient_accumulation_steps': args.ga_steps,
        'gradient_clipping': GRAD_CLIP_NORM,
        'fp16': {
            'enabled': args.fp16,
        },
        'amp': {
            'enabled': args.amp,
            'opt_level': 'O1',
        },
        "flops_profiler": {
            "enabled": args.flops_profiler,
            "profile_step": 200,
            "module_depth": -1,
            "top_modules": 1,
            "detailed": True,
            "output_file": None # TODO Can't get this to work.
        },
        "zero_optimization": {
           "stage": args.zero_stage,
           # Offload the model parameters If you have an nvme drive - you should use the nvme option.
           # Otherwise, use 'cpu' and remove the `nvme_path` line
           # "offload_param": {
           #     "device": "cpu",
           #     # "nvme_path": "/path/to/nvme/folder",
           # },
            # Offload the optimizer of choice. If you have an nvme drive - you should use the nvme option.
            # Otherwise, use 'cpu' and remove the `nvme_path` line
#             "offload_optimizer": {
#                 "device": "cpu",
#                 # "nvme_path": "/path/to/nvme/folder",
#            },
        },
        "optimizer": {
            "type": "Adam",  # You can also use AdamW here
            "params": {
                "lr": LEARNING_RATE,
            },
        },
    }

    if deepspeed_config.get('zero_optimization', {}).get('stage', 0) >= 2:
        print(f"Checkpoints made with DeepSpeed ZeRO Stages 2 and 3 will be stored in deepspeed checkpoint folder")
        print(f"As such, they will require DeepSpeed as a dependency in order to resume from or generate with.")
        print("See the deespeed conversion script for details on how to convert your ZeRO stage 2/3 checkpoint to a single file.")
        print("If using a single GPU, consider running with apex automatic mixed precision instead for a similar speedup to ZeRO.")
        time.sleep(2)

    (distr_dalle, distr_opt, distr_dl, distr_scheduler) = distr_backend.distribute(
        args=args,
        model=dalle,
        optimizer=opt,
        model_parameters=get_trainable_params(dalle),
        training_data=ds,
        # Do not pass the LR scheduler to DeepSpeed so we can manually
        # advance it.
        lr_scheduler=scheduler if LR_DECAY and not using_deepspeed else None,
        config_params=deepspeed_config,
    )

    # training
    for i in range(torch.cuda.device_count()):
        print(f'device {i}:', torch.cuda.get_device_properties(i), flush=True)
    
    with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(
                    wait=1,
                    warmup=1,
                    active=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./profile/train_multinode.prof', worker_name='worker0'),
                record_shapes=True,
                profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
                with_stack=True,
                with_flops=True
        ) as prof:
        for epoch in range(resume_epoch, EPOCHS):
            if data_sampler:
                data_sampler.set_epoch(epoch)
            for i, (text, images) in enumerate(distr_dl):
                if i % 10 == 0 and distr_backend.is_root_worker():
                    t = time.time()
                if args.fp16:
                    images = images.half()
                text, images = map(lambda t: t.cuda(), (text, images))

                loss = distr_dalle(text, images, return_loss=True)

                if using_deepspeed:
                    distr_dalle.backward(loss)
                    distr_dalle.step()
                    # Gradients are automatically zeroed after the step
                else:
                    loss.backward()
                    clip_grad_norm_(distr_dalle.parameters(), GRAD_CLIP_NORM)
                    distr_opt.step()
                    distr_opt.zero_grad()

                # Collective loss, averaged
                avg_loss = distr_backend.average_all(loss)

                log = {}

                if i % 10 == 0 and distr_backend.is_root_worker():
                    print(epoch, i, f'loss - {avg_loss.item()}', flush=True)

                    log = {
                        **log,
                        'epoch': epoch,
                        'iter': i,
                        'loss': avg_loss.item()
                    }

                if i % SAVE_EVERY_N_STEPS == 0:
                    save_model(DALLE_OUTPUT_FILE_NAME, epoch=epoch)

                if i % 100 == 0:
                    if distr_backend.is_root_worker():
                        sample_text = text[:1]
                        token_list = sample_text.masked_select(sample_text != 0).tolist()
                        decoded_text = tokenizer.decode(token_list)

                        if not avoid_model_calls:
                            # CUDA index errors when we don't guard this
                            image = dalle.generate_images(text[:1], filter_thres=0.9)  # topk sampling at 0.9


                        log = {
                            **log,
                        }
                        if not avoid_model_calls:
                            log['image'] = wandb.Image(image, caption=decoded_text)

                if i % 10 == 9 and distr_backend.is_root_worker():
                    sample_per_sec = BATCH_SIZE * 10 / (time.time() - t)
                    log["sample_per_sec"] = sample_per_sec
                    # log['device 0 allocated memory'] = torch.cuda.memory_allocated(0)
                    # log['device 0 reserved memory'] = torch.cuda.memory_reserved(0)
                    print(epoch, i, f'sample_per_sec - {sample_per_sec}', flush=True)

                if i == 201 and args.flops_profiler:
                    raise StopIteration("Profiler has finished running. Stopping training early.")

                if distr_backend.is_root_worker():
                    wandb.log(log)

                prof.step()

            if LR_DECAY:
                distr_scheduler.step(avg_loss)

            save_model(DALLE_OUTPUT_FILE_NAME, epoch=epoch)

            # if distr_backend.is_root_worker():
        #         # save trained model to wandb as an artifact every epoch's end

        #         model_artifact = wandb.Artifact('trained-dalle', type='model', metadata=dict(model_config))
        #         model_artifact.add_file(DALLE_OUTPUT_FILE_NAME)
        #         run.log_artifact(model_artifact)

#     save_model(DALLE_OUTPUT_FILE_NAME, epoch=epoch)
#     if distr_backend.is_root_worker():
    #    wandb.save(DALLE_OUTPUT_FILE_NAME)
    #     model_artifact = wandb.Artifact('trained-dalle', type='model', metadata=dict(model_config))
    #     model_artifact.add_file(DALLE_OUTPUT_FILE_NAME)
    #     run.log_artifact(model_artifact)

    wandb.finish()

    distr_backend.destroy_process_group()
    

if __name__ == "__main__":
    rank = int(os.getenv('SLURM_PROCID'))
    gpus_per_node = int(os.getenv('GPUS_PER_NODE'))
    os.environ["LOCAL_RANK"] = str(rank % gpus_per_node)
    print('local_rank, rank, gpus-per:', os.environ["LOCAL_RANK"], rank, gpus_per_node, flush=True)

    main()
