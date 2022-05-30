#!/bin/bash

time accelerate launch /home/sourab/deepspeed-test/src/language-modeling/run_clm_no_trainer_deepspeed.py \
--config_name "gpt2-large" \
--tokenizer_name "gpt2-large" \
--dataset_name "wikitext" \
--dataset_config_name "wikitext-2-raw-v1" \
--block_size 128 \
--output_dir "/home/sourab/deepspeed-test/glue/clm_deepspeed_stage2_accelerate" \
--learning_rate 5e-4 \
--weight_decay 1e-4 \
--per_device_train_batch_size 30 \
--per_device_eval_batch_size 30 \
--num_train_epochs 3 \
--with_tracking \
