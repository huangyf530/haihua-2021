import transformers
import argparse
import logging
import math
import os
import random
import datasets
from datasets import load_dataset, load_metric
from tqdm.auto import tqdm
import torch
import numpy as np
from torch.utils.data import RandomSampler, SequentialSampler, DistributedSampler
from torch.utils.data.dataloader import DataLoader
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)
# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_model_distributed(args, model):
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    # data args
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--predict_file", type=str, default=None, help="A csv or a json file containing the predict data."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    # model args
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    # train args
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--warmup", type=float, default=0.05, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training"
    )
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument(
        "--tensorboard_dir",
        type=str,
        help="tensorboard dir, if None, no tensorboard."
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="after how many steps to evaluation on dev set. None for no evaluation"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="whether to train the model"
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="whehter to evaluation the model"
    )
    parser.add_argument(
        "--predict",
        type="store_true",
        help="whehter to predict on test dataset."
    )
    args = parser.parse_args()
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args

def train(args, model, train_data, dev_data, device):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=args.n_gpu)
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_data) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(args.max_train_steps * args.warmup),
        num_training_steps=args.max_train_steps,
    )
    start_epoch = 0
    purge_step = None
    if os.path.isdir(args.model_name_or_path):
        start_epoch, completed_steps, args.max_train_steps = load_checkpoint_from_disk(args.model_name_or_path, oprimizer, lr_scheduler)
        purge_step = completed_steps
    if args.local_rank in [-1, 0]:
        if args.tensorboard_dir is not None:
            tb_writer = SummaryWriter(args.tensorboard_dir, purge_step=purge_step)
    # Distributed training (should be after apex fp16 initialization)
    model = set_model_distributed(args, model)
    # Metrics
    metric = load_metric("accuracy")
    # Train!
    world_size = (torch.distributed.get_world_size() if args.local_rank != -1 else 1)
    total_batch_size = args.per_device_train_batch_size * world_size * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_data)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    if os.path.isdir(args.model_name_or_path):
        logger.info(f"  Load checkpoint from {args.model_name_or_path}")
        logger.info(f"  Competed optimization steps = {completed_steps}")
        logger.info(f"  Start epoch = {start_epoch}")
    else:
        completed_steps = 0
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=args.local_rank not in [-1, 0])
    log_loss = 0.0
    accumulate_step = 0
    for epoch in range(start_epoch, args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = (t.to(device) for t in batch)
            input_ids, attention_masks, token_type_ids, labels = batch
            batch_size = input_ids.shape[0]
            sequence_len = input_ids.shape[-1]
            inputs = {"input_ids": input_ids,
                      "attention_mask": attention_masks,
                      "token_type_ids": token_type_ids,
                      "labels": labels}
            outputs = model(**inputs)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            accumulate_step += 1
            loss.backward()
            log_loss += loss.item()
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
                log_loss = log_steps * args.gradient_accumulation_steps / accumulate_step
                if args.local_rank in [-1, 0] and completed_steps % args.log_steps == 0:
                    # log information
                    log_loss = torch.tensor([log_loss], device=device)
                    torch.distributed.all_reduce(log_loss)
                    log_loss = log_loss[0] / torch.distributed.get_world_size()
                    log_loss = log_loss.item()
                    tb_writer.add_scalar('lr', lr_scheduler.get_last_lr()[0], completed_steps)
                    tb_writer.add_scalar('loss/Train', log_loss, completed_steps)
                    progress_bar.set_description("Train loss: {:.4f}".format(log_loss))
                log_loss = 0.0
                accumulate_step = 0
                if args.eval_steps is not None and completed_steps % args.eval_steps == 0:
                    # evaluation during train
                    loss, eval_metric = evaluation(args, model, dev_data, device, metric)
                    if args.local_rank in [-1, 0]:
                        # only main process can log
                        tb_writer.add_scalar('loss/Eval', loss, completed_steps)
                        # TODO: add metric log
                if args.local_rank in [-1, 0] and args.save_steps is not None and completed_steps % args.save_steps == 0:
                    # save the model on main process
                    output_dir = os.path.join(args.output_dir, "checkpoint-{:d}".format(completed_steps))
                    if not os.path.exists(output_dir):
                        os.mkdir(output_dir)
                    save_model(model, optimizer, lr_scheduler, output_dir, epoch, completed_steps, args.max_train_steps)

            if completed_steps >= args.max_train_steps:
                break
    loss, eval_metric = evaluation(args, model, dev_data, device, metric)
    if args.local_rank in [-1, 0]:
        # only main process can log
        tb_writer.add_scalar('loss/Eval', loss, completed_steps)
        # TODO: add metric log
    if args.local_rank in [-1, 0]:
        # save the model on main process
        output_dir = os.path.join(args.output_dir, "checkpoint-{:d}".format(completed_steps))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        save_model(model, optimizer, lr_scheduler, output_dir, epoch, completed_steps, args.max_train_steps)
    return completed_steps

def evaluation(args, model, dev_data, device, metric):
    """evlaluation on dev data or test data"""
    model.eval()
    eval_batch_size = args.per_device_eval_batch_size * max(1, args.n_gpu)
    dev_sampler = SequentialSampler(dev_data) if args.local_rank == -1 else DistributedSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, batch_size=eval_batch_size, sampler=dev_sampler, num_workers=args.n_gpu)
    losses = []
    for index, batch in enumerate(dev_dataloader):
        batch = (t.to(device) for t in batch)
        input_ids, attention_masks, token_type_ids, labels = batch
        batch_size = input_ids.shape[0]
        sequence_len = input_ids.shape[-1]
        inputs = {"input_ids": input_ids,
                    "attention_mask": attention_masks,
                    "token_type_ids": token_type_ids,
                    "labels": labels}
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=-1)
        loss = outputs.loss
        if args.local_rank != -1:
            torch.distributed.all_reduce(loss)
            loss = loss / torch.distributed.get_world_size()
        losses.append(loss.item())
        if args.local_rank != -1:
            all_predictions = [None] * torch.distributed.get_world_size() if args.local_rank == 0 else None
            torch.distributed.gather(predictions, all_predictions, dst=0)
            all_labels = [None] * torch.distributed.get_world_size() if args.local_rank == 0 else None
            torch.distributed.gather(labels, all_labels, dst=0)
            if args.local_rank == 0:
                labels = torch.stack(all_labels, dim=0)
                predictions = torch.stack(all_predictions, dim=0)
        if args.local_rank in [-1, 0]:  
            metric.add_batch(
                predictions=predictions,
                references=labels,
            )
    eval_metric = None
    loss = np.mean(losses)
    if args.local_rank in [-1, 0]:
        eval_metric = metric.compute()
        logger.info(f"Evaluation on {len(dev_data)} examples:  loss {loss} {eval_metric}")
    model.train()  # to trian mode
    if args.local_rank != -1:
        # synchronize
        torch.distributed.barrier()
    return loss, eval_metric

def predict(args, model, test_data, device):
    """predict on test data, should be done on single gpu"""
    model = model.module if hasattr(model, "module") else model
    model.eval()
    eval_batch_size = args.per_device_eval_batch_size
    dev_sampler = SequentialSampler(test_data)
    dev_dataloader = DataLoader(test_data, batch_size=eval_batch_size, sampler=dev_sampler, num_workers=1)
    all_predictions = []
    for index, batch in enumerate(dev_dataloader):
        batch = (t.to(device) for t in batch)
        input_ids, attention_masks, token_type_ids = batch
        batch_size = input_ids.shape[0]
        sequence_len = input_ids.shape[-1]
        inputs = {"input_ids": input_ids,
                    "attention_mask": attention_masks,
                    "token_type_ids": token_type_ids}
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=-1)
        all_predictions.extend(predictions.tolist())
    return all_predictions

def save_model(model, optimizer, scheduler, output_dir, epoch, completed_steps, max_steps):
    model_to_save = model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    checkpoint = {
        'optimizer':optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "completed_steps": completed_steps,
        "max_steps": max_steps,
    }
    torch.save(checkpoint, os.path.join(output_dir, "optimizer.pth"))
    logger.info(f"Save model to {output_dir}")

def load_checkpoint_from_disk(path, optimizer, scheduler):
    checkpoint = torch.load(os.path.join(path, "optimizer.pth"))
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['lr_scheduler'])
    completed_steps = checkpoint['completed_steps']
    epoch = checkpoint['epoch']
    max_steps = checkpoint['max_steps']
    return epoch, completed_steps, max_steps

if __name__=="__main__":
    args = parse_args()
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if args.local_rank in [0, -1] else logging.ERROR)
    if args.local_rank in [-1, 0]:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    extension = args.train_file.split(".")[-1]

    raw_datasets = load_dataset(extension, data_files=data_files)
    
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    if args.model_name_or_path:
        model = AutoModelForMultipleChoice.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMultipleChoice.from_config(config)
    
    if args.local_rank == 0:
        torch.distributed.barrier()
    model.to(device)
    if args.train:
        total_steps = train(args, model, train_data, dev_data, device)
    if args.eval and not args.train:
        # only evaluation without train
        model = set_model_distributed(args, model)
        metric = load_metric("accuracy")
        loss, eval_metric = evaluation(args, model, dev_data, device, metric)
    if args.predict:
        if args.local_rank in [-1, 0]:
            # get model on single gpu
            model = model.module if hasattr(model, "module") else model
            predictions = predict(args, model, test_data， device)
    
