python run_summarization.py \
    --model_name_or_path "/gscratch/argon/artidoro/eventsrealm/real_events/outputs/gold-v1.1-prompt-t5-merged-entities-k_prompts-5-single-all-2/checkpoint-22500" \
    --train_file "/gscratch/argon/artidoro/eventsrealm/real_events/data/gold-v1.1-prompt-t5-merged-entities-k_prompts-1-single-all/train.json" \
    --validation_file "/gscratch/argon/artidoro/eventsrealm/real_events/data/gold-v1.1-prompt-t5-merged-entities-k_prompts-1-single-all/test.json" \
    --do_eval \
    --logging_first_step \
    --output_dir "/gscratch/argon/artidoro/eventsrealm/real_events/outputs/gold-v1.1-prompt-t5-merged-entities-k_prompts-5-single-all-2_test_checkpoint-22500" \
    --predict_with_generate \
    --per_device_train_batch_size 35 \
    --per_device_eval_batch_size 150 \
    --save_total_limit 2 \
    --evaluation_strategy steps \
    --logging_steps 500 \
    --save_steps 500 \
    --eval_steps 500 \
    --load_best_model_at_end  \
    --metric_for_best_model 'different_f1' \
    --remove_unused_columns False\
    --overwrite_output_dir \
    --num_train_epochs 10 \
    --overwrite_cache \
    --learning_rate 5e-5 \
    --text_column input_text \
    --summary_column target_text \
    --label_smoothing_factor 0.1 \

    #--remove_unused_columns False\
    #--source_prefix "question: " \
    # --max_train_samples 100 \
    #--fp16
    #learn rate = 5e-5 for t5-base, 1e-4 for t5-small
    #batch size 8 for t5-base
    