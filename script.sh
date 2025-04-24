export SERVER_ADDR="127.0.0.1"
export SERVER_PORT="8000"
export MODEL_PATH="HF://mlc-ai/Llama-3.1-8B-Instruct-q0f16-MLC" # or the path of other model
export MODEL="Llama-3.1-70B-Instruct-q0f16-MLC" # or other model names
export TOKENIZER="./dist/Llama-3.1-70B-Instruct-q0f16-MLC" # or the path of other tokenizer
export DATA_PATH="./data/dataset"
export N_GPU=4
export ACC_RAW="./data/accuracy_raw"
export ACC_SUM="./data/accuracy_summary"
export EFF="./data/efficiency"



# Launch the server

# Launch mlc-llm server
mlc_llm serve $MODEL_PATH --mode server \
--host $SERVER_ADDR --port $SERVER_PORT --enable-debug --prefix-cache-mode disable

# Or launch sglang server
python -m sglang.launch_server --model-path $MODEL_PATH \
--host $SERVER_ADDR --port $SERVER_PORT --disable-radix-cache  --dtype float16 \
--enable-torch-compile 



# Test accuracy

python accuracy.py --model $MODEL --tokenizer $TOKENIZER \
--dataset BFCL_v3_parallel --num-requests 200 \
--dataset-path $DATA_PATH --num-gpus $N_GPU \
--num-warmup-requests 1 --request-rate inf \
--host $SERVER_ADDR --port $SERVER_PORT --api-endpoint mlc --output $ACC_RAW \
--temperature 0.001 --top-p 0.9 

python accuracy.py --model $MODEL --tokenizer $TOKENIZER \
--dataset BFCL_v3_parallel --num-requests 200 \
--dataset-path $DATA_PATH --num-gpus $N_GPU \
--num-warmup-requests 1 --request-rate inf \
--host $SERVER_ADDR --port $SERVER_PORT --api-endpoint mlc --output $ACC_RAW \
--temperature 0.001 --top-p 0.9 \
--use-stag

python accuracy.py --model $MODEL --tokenizer $TOKENIZER \
--dataset BFCL_v3_live_multiple --num-requests 1052 \
--dataset-path $DATA_PATH --num-gpus $N_GPU \
--num-warmup-requests 1 --request-rate inf \
--host $SERVER_ADDR --port $SERVER_PORT --api-endpoint mlc --output $ACC_RAW \
--temperature 0.001 --top-p 0.9 

python accuracy.py --model $MODEL --tokenizer $TOKENIZER \
--dataset BFCL_v3_live_multiple --num-requests 1052 \
--dataset-path $DATA_PATH --num-gpus $N_GPU \
--num-warmup-requests 1 --request-rate inf \
--host $SERVER_ADDR --port $SERVER_PORT --api-endpoint mlc --output $ACC_RAW \
--temperature 0.001 --top-p 0.9 \
--use-stag

# Process the the output
python check.py --dataset ALL --model ALL --dataset-path $DATA_PATH \
--output-root $ACC_RAW --final-root $ACC_SUM

python draw_accuracy.py --summary-root $ACC_SUM



# Test the efficiency

# With mlc-llm server
python efficiency.py  --model $MODEL --tokenizer $TOKENIZER \
--dataset BFCL_v3_multiple --dataset-path $DATA_PATH --num-gpus $N_GPU \
--num-warmup-requests 200 --num-requests 200 \
--host $SERVER_ADDR --port $SERVER_PORT --num-concurrent-requests 1 \
--api-endpoint mlc --output $EFF --stream \
--temperature 0.001 --top-p 0.9 \
--use-stag

python efficiency.py  --model $MODEL --tokenizer $TOKENIZER \
--dataset BFCL_v3_multiple --dataset-path $DATA_PATH --num-gpus $N_GPU \
--num-warmup-requests 200 --num-requests 200 \
--host $SERVER_ADDR --port $SERVER_PORT --num-concurrent-requests 1 \
--api-endpoint mlc --output $EFF --stream \
--temperature 0.001 --top-p 0.9 

# With sglang server
python efficiency.py  --model $MODEL --tokenizer $TOKENIZER \
--dataset BFCL_v3_multiple --dataset-path ./data/dataset --num-gpus $N_GPU \
--num-warmup-requests 200 --num-requests 200 \
--host $SERVER_ADDR --port $SERVER_PORT --num-concurrent-requests 1 \
--api-endpoint sglang --output $EFF --stream \
--temperature 0.001 --top-p 0.9 \
--use-stag

python efficiency.py  --model $MODEL --tokenizer $TOKENIZER \
--dataset BFCL_v3_multiple --dataset-path ./data/dataset --num-gpus $N_GPU \
--num-warmup-requests 200 --num-requests 200 \
--host $SERVER_ADDR --port $SERVER_PORT --num-concurrent-requests 1 \
--api-endpoint sglang --output $EFF --stream \
--temperature 0.001 --top-p 0.9 

# Process the the output
python check.py --dataset ALL --model ALL --dataset-path $DATA_PATH \
--output-root $ACC_RAW --final-root $ACC_SUM

python draw_efficiency.py --bench-root $ACC_SUM