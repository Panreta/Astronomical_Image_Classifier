# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # Specifies which GPU(s) to use (GPUs with IDs 0 and 1 in this case).
export NPROC_PER_NODE=8 # Sets the number of processes (parallel workers) per node to 2.
export NCCL_P2P_DISABLE=1  # Disables Peer-to-Peer communication
export NCCL_IB_DISABLE=1   # Disables InfiniBand (not applicable for consumer GPUs)

# 运行 SFT 命令
python /home/share/guofangkeda/wangcunshi/ms-swift/swift/cli/sft.py \
  --model_type qwen2_5\
  --model qwen/Qwen2-7B-Instruct \
  --train_type lora \
  --dataset /home/share/guofangkeda/wangcunshi/LSST/jsonl/prompts.jsonl\
  --num_train_epochs 2 \
  --train_type lora \
  --save_total_limit -1 \
  --save_strategy epoch \
  --max_length 4096 \
  --per_device_train_batch_size 4

#  --val_dataset /home/pod/shared-nvme/Cunshi/Xray/val.jsonl \
#  --deepspeed default-zero \
# OpenBMB/MiniCPM-V-2_6  qwen2-vl-7b-instruct

#--model_type 指定模型类型
#--model   # 模型名称（Hugging Face / ModelScope）
#--num_train_epochs 2 \ # Train for 3 full passes over the dataset
#  --dataset_test_ratio 0 \ # no 
#  --save_total_limit -1 \ #keep all checkpoints
#  --save_strategy epoch \ # checkpoint is saved at the end of every training epoch.
#  --max_length 4096 \ # 单样本的tokens最大长度