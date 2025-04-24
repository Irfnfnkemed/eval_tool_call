export SERVER_ADDR="127.0.0.1"
export SERVER_PORT="8000"
export MODEL="Qwen2.5-72B-Instruct-q0f16-MLC"
export TOKENIZER=../Qwen2.5-72B-Instruct-q0f16-MLC
export N_GPU=4
export ACC_RAW="./data/ACC_F"
export ACC_SUM="./data/ACC_SUM_F"
export EFF="./data/EFF_F"

python accuracy.py --model $MODEL --tokenizer $TOKENIZER \
--dataset BFCL_v3_parallel --num-requests 200 \
--dataset-path ./data/dataset --num-gpus $N_GPU \
--num-warmup-requests 1 --request-rate inf \
--host $SERVER_ADDR --port $SERVER_PORT --api-endpoint mlc --output $ACC_RAW \
--temperature 0.001 --top-p 0.9 

python accuracy.py --model $MODEL --tokenizer $TOKENIZER \
--dataset BFCL_v3_parallel --num-requests 200 \
--dataset-path ./data/dataset --num-gpus $N_GPU \
--num-warmup-requests 1 --request-rate inf \
--host $SERVER_ADDR --port $SERVER_PORT --api-endpoint mlc --output $ACC_RAW \
--temperature 0.001 --top-p 0.9 \
--use-stag

python accuracy.py --model $MODEL --tokenizer $TOKENIZER \
--dataset BFCL_v3_live_multiple --num-requests 1052 \
--dataset-path ./data/dataset --num-gpus $N_GPU \
--num-warmup-requests 1 --request-rate inf \
--host $SERVER_ADDR --port $SERVER_PORT --api-endpoint mlc --output $ACC_RAW \
--temperature 0.001 --top-p 0.9 

python accuracy.py --model $MODEL --tokenizer $TOKENIZER \
--dataset BFCL_v3_live_multiple --num-requests 1052 \
--dataset-path ./data/dataset --num-gpus $N_GPU \
--num-warmup-requests 1 --request-rate inf \
--host $SERVER_ADDR --port $SERVER_PORT --api-endpoint mlc --output $ACC_RAW \
--temperature 0.001 --top-p 0.9 \
--use-stag

python efficiency.py  --model $MODEL --tokenizer $TOKENIZER \
--dataset BFCL_v3_multiple --dataset-path ./data/dataset --num-gpus $N_GPU \
--num-warmup-requests 200 --num-requests 200 \
--host $SERVER_ADDR --port $SERVER_PORT --num-concurrent-requests 1 \
--api-endpoint mlc --output $EFF --stream \
--temperature 0.001 --top-p 0.9 \
--use-stag

python efficiency.py  --model $MODEL --tokenizer $TOKENIZER \
--dataset BFCL_v3_multiple --dataset-path ./data/dataset --num-gpus $N_GPU \
--num-warmup-requests 200 --num-requests 200 \
--host $SERVER_ADDR --port $SERVER_PORT --num-concurrent-requests 1 \
--api-endpoint mlc --output $EFF --stream \
--temperature 0.001 --top-p 0.9 
