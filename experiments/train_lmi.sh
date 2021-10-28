# name from filename
NAME=$0
NAME=${NAME##*/}
NAME=${NAME%.*}
NAME=${NAME#train_}

python full_train.py \
    --name ${NAME}  \
    --desc ${NAME}  \
    --indir 'datasets/ABC' \
    --outdir 'models'  \
    --logdir 'logs' \
    --trainset 'trainset.txt'  \
    --testset 'testset.txt'  \
    --nepoch 120 \
    --save_interval 10\
    --lr 0.01  \
    --scheduler_steps 75 \
    --debug 0  \
    --workers 22  \
    --batchSize 700  \
    --points_per_patch 200  \
    --patches_per_shape 500  \
    --sub_sample_size 1000  \
    --cache_capacity 30  \
    --patch_radius 0.0  \
    --w 4 \
    --k 10 \
    --grid_resolution 256 \
    --training_order 'random_shape_consecutive'  \

