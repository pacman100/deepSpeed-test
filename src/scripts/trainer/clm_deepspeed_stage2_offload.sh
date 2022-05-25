#!/bin/bash

time torchrun --nproc_per_node=2 /home/sourab/deepspeed-test/src/language-modeling/run_clm.py \
--config_name "gpt2-large" \
--tokenizer_name "gpt2-large" \
--dataset_name "wikitext" \
--dataset_config_name "wikitext-2-raw-v1" \
--block_size 128 \
--output_dir "/home/sourab/deepspeed-test/glue/clm_deepspeed_stage2_offload_trainer" \
--overwrite_output_dir \
--do_train \
--do_eval \
--evaluation_strategy "epoch" \
--per_device_train_batch_size 40 \
--per_device_eval_batch_size 40 \
--learning_rate 5e-4 \
--logging_steps 100 \
--save_strategy "epoch" \
--save_total_limit 1 \
--fp16 \
--fp16_full_eval \
--optim "adamw_torch" \
--report_to "wandb" \
--deepspeed "/home/sourab/deepspeed-test/src/configs/zero2_offload_config.json" \
