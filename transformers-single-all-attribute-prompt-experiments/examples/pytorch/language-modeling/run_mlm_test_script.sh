python run_mlm.py \
    --model_name_or_path roberta-large \
    --validation_file "/usr0/home/espiliop/pet/real_events/data/gold-v1.1-prompt-mlm-merged-entities-simplified/test.json" \
    --do_eval \
    --logging_first_step \
    --output_dir "/usr0/home/espiliop/pet/real_events/outputs/prompt-mlm-test-zeroshot" \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 10 \
    --remove_unused_columns False\
    --save_total_limit 1 \
    --evaluation_strategy steps \
    --logging_steps 1000 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --fp16 \
    --overwrite_output_dir \
    --max_seq_length 512 \
    # --max_eval_samples 500 \
    # --max_train_samples 100 \
    # --overwrite_cache \