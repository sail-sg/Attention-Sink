export WANDB_PROJECT=YOUR_PROJECT_NAME
export WANDB_ENTITY=YOUR_WANDB_ENTITY
export WANDB_API_KEY=YOUR_WANDB_API_KEY

export NUMBER_OF_GPUS=8
export MODEL_SIZE=60
export KERNEL=$1
export MODEL_NAME=tinyllama'_'$MODEL_SIZE'M_'$KERNEL
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
    --data_yaml_file configs/attnkernel/tinyllama_$MODEL_SIZE'_'$KERNEL.yaml \
    --out_name $MODEL_NAME \
    --resume True \
    --mode attnkernel \
    --downsample_ratio 1.0

python pretrain/eval_sink_pretrain.py --model_name tinyllama_$MODEL_SIZE'_'$KERNEL --mode attnkernel --load_from checkpoints/$MODEL_NAME/iter-020000-ckpt.pth