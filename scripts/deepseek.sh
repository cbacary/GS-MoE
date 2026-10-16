TESTS_OUTPUT_DIR="$TMPDIR/compressed_output"

# MODEL="deepseek-ai/DeepSeek-V2-Lite" # May require edit of patchers/deepseekv2_rebuild.py to work for V2. Just mirror how it is done in deepseek_rebuild.py
MODEL="deepseek-ai/deepseek-moe-16b-base"
TESTS_OUTPUT_DIR="$TMPDIR/compressed_output"
LAYERS_PER_STEP=14
N_SHARDS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)

NYS_SMOOTHING=5.0
FACT_SMOOTHING=10 
Ex_ridge=1.0
 
# Amount to compress expert parameters to achieve target model compression amount
#                       DS-MoE16B	Qwen2-57B-A14B	Mixtral-8x7B
# 60%:   	        65.73%          69.83% 	        62.14%
# 40%:	                43.82% 	        46.56% 	        41.42%
# 20%:	                21.91% 	        23.28% 	        20.71% 

# Does ~ half factorize / half intermediate width reduction
# Compresses to 60% reduced total model params; 62.14% reduced expert params
for nys_fact_r in 0.55; do
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
                --nys_to_fact_ratio $nys_fact_r \
                --nystrom_ridge 0.1 \
                --Ex_ridge $Ex_ridge \
                --n_shared_projs 8 \
                --dataset "alpaca" \
                --compressed_acts_save_path "$TMPDIR/activations" \
                --save \
                --benchmark
done         


for nys_fact_r in 0.65; do
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
                --compression_ratio 0.4382 \
                --nystrom_layer_smoothing $NYS_SMOOTHING \
                --factorization_layer_smoothing $FACT_SMOOTHING \
                --nys_to_fact_ratio $nys_fact_r \
                --nystrom_ridge 0.1 \
                --Ex_ridge $Ex_ridge \
                --n_shared_projs 2 \
                --dataset "alpaca" \
                --save \
                --benchmark \
                --compressed_acts_save_path "$TMPDIR/activations"
done         


for nys_fact_r in 0.75; do
        python -m src.experiments.run_compression_parallel \
                --shards $N_SHARDS \
                --model "$MODEL" \
                --tokenizer_src "$MODEL" \
                --device 0 \
                --calib_size 512 \
                --calibs_batch_size 32 \
                --layers_per_compression_step $LAYERS_PER_STEP \
                --output_dir "$TESTS_OUTPUT_DIR" \
                --note "leverage rank allocation" \
                --compression_ratio 0.2191 \
                --nystrom_layer_smoothing $NYS_SMOOTHING \
                --factorization_layer_smoothing $FACT_SMOOTHING \
                --nys_to_fact_ratio $nys_fact_r \
                --nystrom_ridge 0.1 \
                --Ex_ridge $Ex_ridge \
                --n_shared_projs 1 \
                --dataset "alpaca" \
                --save \
                --benchmark \
                --compressed_acts_save_path "$TMPDIR/activations"
done         



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
        --nys_to_fact_ratio 0.85 \
        --nystrom_ridge 0.01 \
        --Ex_ridge $Ex_ridge \
        --uniformity "leverage" \
        --dataset "alpaca" \
        --n_shared_projs 1 \

