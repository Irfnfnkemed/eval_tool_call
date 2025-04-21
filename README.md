# Evaluate the tool-calling accuracy and efficiency on LLM engine with Structural Tag

The evaluation script is modified based on the MLC-LLM bench and BFCL ast checker. The script uses the Structural Tag API to test the tool-calling accuracy and efficiency

## Test the accuracy

First launch the server.
```bash
mlc_llm serve HF://mlc-ai/Llama-3.1-8B-Instruct-q0f16-MLC --mode server \
--host 127.0.0.1 --port 8000 --enable-debug 
```

Than generate the raw data (w/ & w/o structural tag):
```bash
cd ./tool_call_eval
python accuracy.py --model Llama-3.1-8B-Instruct-q0f16-MLC \
--tokenizer /dist/Llama-3-8B-Instruct-q0f32-MLC \
--dataset BFCL_v3_simple --dataset-path ./data/dataset --num-gpus 1 \
--num-requests 400 --num-warmup-requests 1 --request-rate inf \
--host 127.0.0.1 --port 8000 \
--api-endpoint mlc --output ./data/accuracy_raw \
--use-jf [--use-stag]
```

The raw data will be in `./data/accuracy_raw` directory. Finally process the raw data:
```bash
python check.py --dataset ALL --model ALL --dataset-path ./data/dataset \
--output-root ./data/accuracy_raw --final-root ./data/accuracy_summary
```

The summary wii be in `./data/accuracy_summary` directory. Run the script to generate summary pictures:
```bash
python draw_accuracy.py --summary-root ./data/accuracy_summary
```

The pictures will be in `./data/accuracy_summary` directory. To get more detailed picture, please run
```bash
python ./draw_accuracy_detail.py --summary-root ./data/accuracy_summary
```

The detailed pictures will also be in `./data/accuracy_summary` directory. 

Note: you may need to modify `SUPPORTED_MODEL` and `SUPPORTED_DATASET` in `check.py`, as well as `models` and  `datasets` in the draw scripts accoring to the specific cases.

## Test the efficiency

Also first launch the server.
```bash
mlc_llm serve HF://mlc-ai/Llama-3.1-8B-Instruct-q0f16-MLC --mode server \
--host 127.0.0.1 --port 8000 --enable-debug 

python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
--host 127.0.0.1 --port 30000 
```

Than generate the raw data (w/ & w/o structural tag, w/ & w/o jump-forward-decoding, mlc/sglang backend):

```bash
python efficiency.py --model Llama-3.1-8B-Instruct-q0f16-MLC \
--tokenizer /dist/Llama-3.1-8B-Instruct-q0f16-MLC \
--dataset BFCL_v3_multiple --dataset-path ./data/dataset --num-gpus 1 \
--num-warmup-requests 1 --num-requests 200 \
--host 127.0.0.1 --port 8000 --num-concurrent-requests 1 \
--api-endpoint [mlc|sglang] --output ./data/efficiecy  \
--stream [--use-stag] [--use-jf] 
```

The bench data will be in `./data/efficiecy` directory. Run the script to generate summary pictures:
```bash
python draw_efficiency.py --bench-root ./data/efficiency
```

The pictures will be in `./data/efficiency` directory. 

Note: you may need to modify `models` and  `datasets`, as well as the desired metrics in `query_to_title` in the draw scripts accoring to the specific cases.