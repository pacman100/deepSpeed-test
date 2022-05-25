#!/bin/bash

time accelerate launch /home/sourab/deepspeed-test/src/text-classification/run_glue_no_trainer.py \
--task_name "mrpc" \
--max_length 128 \
--model_name_or_path "microsoft/deberta-v2-xlarge-mnli" \
--output_dir "/home/sourab/deepspeed-test/glue/mrpc_deepspeed_stage3_accelerate" \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--gradient_accumulation_steps 1 \
--learning_rate 3.5e-6 \
--weight_decay 0.0 \
--max_grad_norm 1.0 \
--num_train_epochs 6 \
--num_warmup_steps 50 \
--with_tracking \
