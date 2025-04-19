# Evaluate the tool-calling accuracy and efficiency on MLC-LLM

The evaluation script is modified based on the MLC-LLM bench and BFCL ast checker.

## Test the accuracy

First launch the server.
```bash
mlc_llm serve HF://mlc-ai/Llama-3-8B-Instruct-q0f16-MLC --mode server --enable-debug
```

Than generate the raw data (w/ & w/o structural tag):
```bash
cd ./tool_call_eval
python accuracy.py --tokenizer /dist/Llama-3-8B-Instruct-q0f32-MLC \
--dataset BFCL_v3_simple --dataset-path ./data/dataset --num-gpus 1 \
--num-requests 400 --num-warmup-requests 1 --request-rate inf --host 127.0.0.1 --port 8000 \
--api-endpoint openai-chat --output ./data/accuracy_raw --request-rate inf --use-jf [--use-stag]
```

The raw data will be in `./data/accuracy_raw` directory. Finally process the raw data:
```bash
python ./new_bench/check.py --dataset ALL --model ALL --dataset-path ./data/dataset \
--output-root ./data/accuracy_raw --final-root ./data/accuracy_summary
```

The summary wii be in `./data/accuracy_summary` directory. Run the script to generate summary pictures:
```bash
python ./draw_display.py --summary-root ./data/accuracy_summary
```

The pictures will be in `./data/accuracy_summary` directory. To get more detailed picture, please run
```bash
python ./draw.py --summary-root ./data/accuracy_summary
```

## Test the efficiency

Also first launch the server.
```bash
mlc_llm serve HF://mlc-ai/Llama-3-8B-Instruct-q0f16-MLC --mode server --enable-debug
```

Than generate the raw data (w/ & w/o structural tag, w/ & w/o jump-forward-decoding):

```bash
python ./efficiency.py --tokenizer /dist/Llama-3-8B-Instruct-q0f32-MLC \
--dataset BFCL_v3_multiple --dataset-path ./data/dataset \
--num-gpus 1 --num-warmup-requests 1 --num-requests 200 --host 127.0.0.1 --port 8000 \
--api-endpoint openai-chat --output ./efficiecy --num-concurrent-requests 1 \
--stream [--use-stag] [--use-jf] 
```

