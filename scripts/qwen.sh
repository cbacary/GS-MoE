TESTS_OUTPUT_DIR="$TMPDIR/compressed_output"

MODEL="Qwen/Qwen2-57B-A14B"
TESTS_OUTPUT_DIR="$TMPDIR/compressed_output"
LAYERS_PER_STEP=1
N_SHARDS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)

NYS_FACT_RATIO=0.55
NYS_SMOOTHING=5.0
FACT_SMOOTHING=10
Ex_ridge=1.0

# Amount to compress expert parameters to achieve target model compression amount
#               DS-MoE16B	Qwen2-57B-A14B	Mixtral-8x7B
# 60%:   	65.73%          69.83% 	        62.14%
# 40%:	        43.82% 	        46.56% 	        41.42%
# 20%:	        21.91% 	        23.28% 	        20.71% 

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
                --compression_ratio 0.6983 \
                --nystrom_layer_smoothing $NYS_SMOOTHING \
                --factorization_layer_smoothing $FACT_SMOOTHING \
                --nys_to_fact_ratio $nys_fact_r \
                --nystrom_ridge 0.1 \
                --Ex_ridge $Ex_ridge \
                --n_shared_projs 8 \
                --dataset "alpaca" \
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
                --compression_ratio 0.4656 \
                --nystrom_layer_smoothing $NYS_SMOOTHING \
                --factorization_layer_smoothing $FACT_SMOOTHING \
                --nys_to_fact_ratio $nys_fact_r \
                --nystrom_ridge 0.1 \
                --Ex_ridge $Ex_ridge \
                --n_shared_projs 2 \
                --dataset "alpaca" \
                --save \
                --benchmark
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
                --note "benchmark" \
                --compression_ratio 0.2328 \
                --nystrom_layer_smoothing $NYS_SMOOTHING \
                --factorization_layer_smoothing $FACT_SMOOTHING \
                --nys_to_fact_ratio $nys_fact_r \
                --nystrom_ridge 0.1 \
                --Ex_ridge $Ex_ridge \
                --n_shared_projs 1 \
                --dataset "alpaca" \
                --save \
                --benchmark
done

