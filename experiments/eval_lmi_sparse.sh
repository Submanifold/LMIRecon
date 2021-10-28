# name from filename
NAME=$0
NAME=${NAME##*/}
NAME=${NAME%.*}
NAME=${NAME#eval_}

python full_eval.py \
    --indir 'datasets' \
    --outdir 'results' \
    --modeldir 'models' \
    --dataset  'GRSI_sparse/testset.txt'\
    --models ${NAME} \
    --modelpostfix '_model_exp.pth' \
    --batchSize 500 \
    --workers 7 \
    --cache_capacity 5 \
    --query_grid_resolution 256 \
    --epsilon 8 \
    --certainty_threshold 32 \
    --sigma 19 \
    --w 4 \
