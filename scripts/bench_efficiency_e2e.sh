BUDGET_POOL=('32' '64' '128' '256' '512' '1024' '2048' '4096' '144880')
CONTEXT_POOL=('10240' '32768' '102400')

for context in "${CONTEXT_POOL[@]}"
do
    for budget in "${BUDGET_POOL[@]}"
    do
        # nsys profile --output $context-$budget \
        # python3 bench_textgen.py --context_len 10240 --decode_len 256 --token_budget 32 --iteration 3
        python3 bench_textgen.py --context_len $context --decode_len 256 --token_budget $budget --iteration 3
    done
done