devices=$1

for subset in 'general-knowledge' 'math-reasoning'
do
    # small-sized open-source models (to be fine-tuned)
    for model in 'meta-llama/Llama-3.1-8B-Instruct' 'Qwen/Qwen3-4B-Instruct-2507' 'mistralai/Mistral-7B-Instruct-v0.3'
    do
        for split in 'train' 'validation' 'test'
        do
            CUDA_VISIBLE_DEVICES=$devices PYTHONPATH=. python run/generate_responses.py \
                --subset $subset \
                --model-name $model \
                --split $split
        done
    done

    # medium-sized open-source models (not fine-tuned)
    for model in 'meta-llama/Llama-3.1-70B-Instruct' 'Qwen/Qwen3-30B-A3B-Instruct-2507' 'mistralai/Mistral-Small-24B-Instruct-2501'
    do
        CUDA_VISIBLE_DEVICES=$devices PYTHONPATH=. python run/generate_responses.py \
            --subset $subset \
            --model-name $model \
            --split test
    done

    # proprietary models & open-source models accessed via API
    for model in 'gemini-2.5-flash-lite' 'claude-3-5-sonnet' 'llama3.1-405b' 'deepseek-r1'
    do
        PYTHONPATH=. python run/generate_responses.py \
            --subset $subset \
            --model-name $model \
            --split test
    done    
done
