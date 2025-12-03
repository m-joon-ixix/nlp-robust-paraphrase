devices=$1

for subset in 'general-knowledge' 'math-reasoning'
do
    for model in 'meta-llama/Llama-3.1-8B-Instruct' 'Qwen/Qwen3-4B-Instruct-2507' 'mistralai/Mistral-7B-Instruct-v0.3'
    do
        CUDA_VISIBLE_DEVICES=$devices PYTHONPATH=. python training/sft.py \
            --model-name $model \
            --subset $subset
    done
done
