#!/bin/bash

time torchrun --nproc_per_node=2 /home/sourab/deepspeed-test/src/text-classification/run_glue.py \
--task_name "mrpc" \
--max_seq_len 128 \
--model_name_or_path "microsoft/deberta-v2-xlarge-mnli" \
--output_dir "/home/sourab/deepspeed-test/glue/mrpc_deepspeed_stage1_trainer" \
--overwrite_output_dir \
--do_train \
--evaluation_strategy "epoch" \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--gradient_accumulation_steps 1 \
--learning_rate 3.5e-6 \
--weight_decay 0.0 \
--max_grad_norm 1.0 \
--num_train_epochs 6 \
--lr_scheduler_type "linear" \
--warmup_steps 50 \
--logging_steps 100 \
--fp16 \
--fp16_full_eval \
--optim "adamw_torch" \
--report_to "wandb" \
--deepspeed "/home/sourab/deepspeed-test/src/configs/zero1_config.json"
