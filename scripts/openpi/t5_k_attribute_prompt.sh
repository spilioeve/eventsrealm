python transformers-single-all-prompt-experiments/examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path t5-base \
    --train_file "data/preprocessed_openpi_k_attribute_data/train.json" \
    --validation_file "data/preprocessed_openpi_k_attribute_data/dev.json" \
    --do_train \
    --do_eval \
    --logging_first_step \
    --output_dir "outputs/k_attribute" \
    --predict_with_generate \
    --per_device_train_batch_size 40 \
    --per_device_eval_batch_size 40 \
    --save_total_limit 2 \
    --evaluation_strategy steps \
    --logging_steps 1000 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --load_best_model_at_end  \
    --metric_for_best_model 'different_f1' \
    --remove_unused_columns False\
    --overwrite_output_dir \
    --num_train_epochs 8 \
    --overwrite_cache \
    --learning_rate 5e-5 \
    --text_column input_text \
    --summary_column target_text \
    --label_smoothing_factor 0.1 \