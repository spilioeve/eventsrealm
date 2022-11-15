deepspeed \
run_mlm.py \
    --model_name_or_path roberta-large \
    --train_file "/usr0/home/espiliop/pet/real_events/data/gold-v1.1-prompt-mlm/train.json" \
    --validation_file "/usr0/home/espiliop/pet/real_events/data/gold-v1.1-prompt-mlm/dev.json" \
    --do_train \
    --do_eval \
    --logging_first_step \
    --output_dir "/usr0/home/espiliop/pet/real_events/outputs/prompt-mlm/" \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --save_total_limit 1 \
    --evaluation_strategy steps \
    --logging_steps 1000 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --fp16 \
    --overwrite_output_dir \
    --max_seq_length 512 \
    --deepspeed deepspeed_config.json \
    # --max_train_samples 100 \
    # --max_eval_samples 100 \
    # --overwrite_cache \