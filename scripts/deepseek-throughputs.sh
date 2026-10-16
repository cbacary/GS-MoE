TESTS_OUTPUT_DIR="$TMPDIR/compressed_output"

# MODEL="deepseek-ai/DeepSeek-V2-Lite" # Should also work for V2.
MODEL="deepseek-ai/deepseek-moe-16b-base"
TESTS_OUTPUT_DIR="$TMPDIR/compressed_output"
LAYERS_PER_STEP=14
N_SHARDS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)

# NYS_FACT_RATIO=0.5503988681
NYS_FACT_RATIO=0.55 # BEST DEEPSEEK @ 0.60 compression ratio
# NYS_SMOOTHING=50.000
NYS_SMOOTHING=5.0 # BEST DEEPSEEK
# FACT_SMOOTHING=0.13746
# FACT_SMOOTHING=30.0
FACT_SMOOTHING=10 
Ex_ridge=1.0

#               DS-MoE16B	Qwen2-57B-A14B	Mixtral-8x7B
# 60%:   	65.73%          69.83% 	        62.14%
# 40%:	        43.82% 	        46.56% 	        41.42%
# 20%:	        21.91% 	        23.28% 	        20.71% 

# Does ~ half factorize / prune then ALL fact ALL prune
# Compresses to 60% reduced total model params; 62.14% reduced expert params
for shared_projs in 8 4 2 1; do
    python -m src.experiments.run_compression_parallel \
            --shards $N_SHARDS \
            --model "$MODEL" \
            --tokenizer_src "$MODEL" \
            --device 0 \
            --calib_size 512 \
            --calibs_batch_size 32 \
            --layers_per_compression_step $LAYERS_PER_STEP \
            --output_dir "$TESTS_OUTPUT_DIR" \
            --note "benchmark" \
            --compression_ratio 0.6573 \
            --nystrom_layer_smoothing $NYS_SMOOTHING \
            --factorization_layer_smoothing $FACT_SMOOTHING \
            --nys_to_fact_ratio 0.55 \
            --nystrom_ridge 0.1 \
            --Ex_ridge $Ex_ridge \
            --n_shared_projs $shared_projs \
            --dataset "alpaca" \
            --compressed_acts_save_path "$TMPDIR/activations" \
            --save

    python -m src.experiments.run_throughput \
        --shards 1 \
        --model $TESTS_OUTPUT_DIR/model/ \
        --tokenizer_src "$MODEL" \
        --device 0 \
        --calib_size 512 \
        --calibs_batch_size 32 \
        --layers_per_compression_step $LAYERS_PER_STEP \
        --output_dir "$TESTS_OUTPUT_DIR" \
        --note "throughput" \
        --compression_ratio  0.6214 \
        --nystrom_layer_smoothing $NYS_SMOOTHING \
        --factorization_layer_smoothing $FACT_SMOOTHING \
        --nys_to_fact_ratio 0.55 \
        --nystrom_ridge 0.01 \
        --Ex_ridge $Ex_ridge \
        --uniformity "leverage" \
        --dataset "alpaca" \
        --n_shared_projs $shared_projs
done         


