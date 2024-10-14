export WANDB_PROJECT=YOUR_PROJECT_NAME
export WANDB_ENTITY=YOUR_WANDB_ENTITY
export WANDB_API_KEY=YOUR_WANDB_API_KEY

export NUMBER_OF_GPUS=8
export MODEL_SIZE='1b'
export MODEL_NAME=tinyllama'_'$MODEL_SIZE
export WANDB_NAME=$MODEL_NAME

lightning run model \
    --node-rank=0  \
    --main-address=127.0.0.1 \
    --accelerator=cuda \
    --num-nodes=1 \
    --devices=$NUMBER_OF_GPUS \
    pretrain/tinyllama.py --devices $NUMBER_OF_GPUS \
    --train_data_dir datasets/lit_dataset_regmix \
    --val_data_dir datasets/lit_dataset_regmix \
    --data_yaml_file configs/tinyllama_$MODEL_SIZE'_sigmoid'.yaml \
    --out_name $MODEL_NAME \
    --resume True \
    --mode attnsim \
    --downsample_ratio 1.0

python pretrain/eval_sink_pretrain.py --model_name tinyllama_1_1b_sim_sigmoid_no_norm --mode attnsim --load_from checkpoints/$MODEL_NAME/iter-040000-ckpt.pth