MODEL="mistralai/Mixtral-8x7B-v0.1"

TESTS_OUTPUT_DIR="$TMPDIR/compressed_output"

LAYERS_PER_STEP=4 # Lower if GPU not big enough | if GPU barely fits model you will have to refactor activations to store on cpu
N_SHARDS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)

TESTS_OUTPUT_DIR="$TMPDIR/compressed_output"

NYS_SMOOTHING=10.0
FACT_SMOOTHING=5
Ex_ridge=1.0

# Amount to compress expert parameters to achieve target model compression amount
#               DS-MoE16B	Qwen2-57B-A14B	Mixtral-8x7B
# 60%:   	65.73%          69.83% 	        62.14%
# 40%:	        43.82% 	        46.56% 	        41.42%
# 20%:	        21.91% 	        23.28% 	        20.71% 


# Does split between factorize and intermediate width reduction
# Compresses to 60% reduced total model params; 62.14% reduced expert params
for nys_fact_r in 0.85; do
        python -m src.experiments.run_compression_parallel \
                --shards $N_SHARDS \
                --model "$MODEL" \
                --tokenizer_src "$MODEL" \
                --device 0 \
                --calib_size 512 \
                --calibs_batch_size 32 \
                --layers_per_compression_step $LAYERS_PER_STEP \
                --output_dir "$TESTS_OUTPUT_DIR" \
                --note "Benchmarks" \
                --compression_ratio  0.6214 \
                --nystrom_layer_smoothing $NYS_SMOOTHING \
                --factorization_layer_smoothing $FACT_SMOOTHING \
                --nys_to_fact_ratio $nys_fact_r \
                --nystrom_ridge 0.01 \
                --Ex_ridge $Ex_ridge \
                --dataset "alpaca" \
                --compressed_acts_save_path "$TMPDIR/activations" \
                --n_shared_projs 1 \
                --save \
                --benchmark
done

for nys_fact_r in 0.85; do
        python -m src.experiments.run_compression_parallel \
                --shards $N_SHARDS \
                --model "$MODEL" \
                --tokenizer_src "$MODEL" \
                --device 0 \
                --calib_size 512 \
                --calibs_batch_size 32 \
                --layers_per_compression_step $LAYERS_PER_STEP \
                --output_dir "$TESTS_OUTPUT_DIR" \
                --note "Ratio" \
                --compression_ratio  0.4142 \
                --nystrom_layer_smoothing $NYS_SMOOTHING \
                --factorization_layer_smoothing $FACT_SMOOTHING \
                --nys_to_fact_ratio $nys_fact_r \
                --nystrom_ridge 0.01 \
                --Ex_ridge $Ex_ridge \
                --dataset "alpaca" \
                --n_shared_projs 1 \
                --benchmark \
                --save \
                --compressed_acts_save_path "$TMPDIR/activations"
done

for nys_fact_r in 0.85; do
        python -m src.experiments.run_compression_parallel \
                --shards $N_SHARDS \
                --model "$MODEL" \
                --tokenizer_src "$MODEL" \
                --device 0 \
                --calib_size 512 \
                --calibs_batch_size 32 \
                --layers_per_compression_step $LAYERS_PER_STEP \
                --output_dir "$TESTS_OUTPUT_DIR" \
                --note "Ratio" \
                --compression_ratio  0.219 \
                --nystrom_layer_smoothing $NYS_SMOOTHING \
                --factorization_layer_smoothing $FACT_SMOOTHING \
                --nys_to_fact_ratio $nys_fact_r \
                --nystrom_ridge 0.01 \
                --Ex_ridge $Ex_ridge \
                --dataset "alpaca" \
                --n_shared_projs 1 \
                --benchmark \
                --save \
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
