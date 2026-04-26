#!/usr/bin/env bash
# DPO with W&B logging
set -euo pipefail
IFS=$'\n\t'

# -------- Increase file descriptor limit --------
ulimit -n 65536

# -------- paths / model --------
cache_dir="/data/home/animesh/experiments/auxdpo/output/cache"
model_name="meta-llama/Llama-3.1-8B"
output_dir="/data/home/animesh/experiments/auxdpo/output/auxdpo_mmlu_pro"
RESUME_CKPT=""

# -------- hardware hint --------
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# -------- conda env --------
env_name='dpo'
eval "$(conda shell.bash hook)"
conda activate "$env_name"

# -------- W&B setup --------
wandb_project="h200_dpo_mmlu_pro_experiments_eval"
export WANDB_PROJECT="$wandb_project"
export WANDB_ENTITY="animeshmadaan"
export WANDB_MODE=online
export WANDB_RUN_GROUP="dpo_mmlu_pro"
export WANDB_TAGS="dpo,auxxpo,mmlu-pro,llama-3.1-8b"
export WANDB_DIR="$output_dir/wandb"

# HuggingFace token (required for gated models like LLaMA)
export HF_TOKEN="${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null || echo '')}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# optional: local cache dirs
export TRANSFORMERS_CACHE="$cache_dir/transformers"
export HF_HOME="$cache_dir/hf_home"
export HF_DATASETS_CACHE="$cache_dir/datasets"

# -------- DPO specific --------
export BETA=0.5

# -------- run controls --------
which_exp=${1:--1}
dryrun=false
debug=false

# hparams
gradient_accumulation_steps=4
batch_size=8
mini_batch_size=2
num_train_epochs=4
lrs=(5e-6)
betas=(0.5)
# χPO is robust at much smaller β (paper uses β∈{5e-3, 1e-2}); scale the DPO β by ~1/100
# when xpo=true. Paper: Huang et al. "Correcting the Mythos of KL-Regularization", 2024.
xpo_beta_scale=0.01
seeds=(0)

# datasets
mmlu_pro='TIGER-Lab/MMLU-Pro'
preference_dataset_paths=("$mmlu_pro")

ipo_loss=false
xpo=true
xpo_r_max=1.0
auxdpo=true
auxdpo_lambda_null=1.0
auxdpo_lambda_amp=0.01
auxdpo_delta_cap=1.0
auxdpo_aux_lr=5e-3

# LoRA config
use_lora=true
lora_r=128
lora_alpha=256
lora_dropout=0.05
lora_target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

exp_num=0

if [[ "$debug" == true ]]; then
  export WANDB_MODE=offline
fi

# -------- choose pretrained_dir and ref_model_path --------
use_model="$model_name"
ref_model_path="$model_name"  # By default, same as training model

if [[ -n "$RESUME_CKPT" ]]; then
  use_model="$RESUME_CKPT"
  ref_model_path="$model_name"  # Keep ref model at base model when resuming
  echo ">> Resuming training from: $use_model"
  echo ">> Using reference model: $ref_model_path"
fi

# -------- loop --------
for preference_dataset_path in "${preference_dataset_paths[@]}"; do
  for beta in "${betas[@]}"; do
    for lr in "${lrs[@]}"; do
      for seed in "${seeds[@]}"; do

        # When χPO is enabled, reduce β by orders of magnitude (paper uses β≪DPO's).
        if [[ "$xpo" == true ]]; then
          effective_beta=$(python -c "print($beta * $xpo_beta_scale)")
        else
          effective_beta=$beta
        fi

        if [[ $which_exp -ge 0 && $exp_num -ne $which_exp ]]; then
          exp_num=$((exp_num+1))
          continue
        fi

        dataset_basename="$(basename -- "$preference_dataset_path")"
        lora_tag=""
        if [[ "$use_lora" == true ]]; then
          lora_tag="_lora_r${lora_r}"
        fi
        algo_tag="dpo"
        if [[ "$xpo" == true ]]; then algo_tag="xpo"; fi
        if [[ "$auxdpo" == true ]]; then algo_tag="aux${algo_tag}"; fi
        run_name="${algo_tag}${lora_tag}_${dataset_basename}_beta${effective_beta}_lr${lr}_bs${batch_size}_gradacc${gradient_accumulation_steps}_seed${seed}"
        echo "Running experiment $exp_num: $run_name"

        command="python -m trainers.dpo \
          --wandb_project \"$wandb_project\" \
          --run_name \"$run_name\" \
          --inner_iteration_steps 1 \
          --batch_size $batch_size \
          --mini_batch_size $mini_batch_size \
          --num_train_epochs $num_train_epochs \
          --pretrained_dir \"$use_model\" \
          --ref_model_path \"$ref_model_path\" \
          --preference_dataset_path \"$preference_dataset_path\" \
          --preference_num_samples -1 \
          --beta $effective_beta \
          --gradient_accumulation_steps $gradient_accumulation_steps \
          --cache_dir \"$cache_dir\" \
          --learning_rate $lr \
          --output_dir \"$output_dir\" \
          --seed $seed \
          --tokenizer_type \"$model_name\" \
          --bf16 True"

        if [[ "$ipo_loss" == true ]]; then
          command+=" --ipo_loss"
        fi

        if [[ "$xpo" == true ]]; then
          command+=" --xpo"
          command+=" --xpo_r_max $xpo_r_max"
        fi

        if [[ "$auxdpo" == true ]]; then
          command+=" --auxdpo"
          command+=" --auxdpo_lambda_null $auxdpo_lambda_null"
          command+=" --auxdpo_lambda_amp $auxdpo_lambda_amp"
          command+=" --auxdpo_delta_cap $auxdpo_delta_cap"
          command+=" --auxdpo_aux_lr $auxdpo_aux_lr"
        else
          command+=" --noauxdpo"
        fi

        if [[ "$use_lora" == true ]]; then
          command+=" --use_lora"
          command+=" --lora_r $lora_r"
          command+=" --lora_alpha $lora_alpha"
          command+=" --lora_dropout $lora_dropout"
          command+=" --lora_target_modules $lora_target_modules"
        else
          command+=" --nouse_lora"
        fi

        echo -e "$command\n"
        if [[ "$dryrun" == false ]]; then
          eval "$command"
          sleep 20
        fi
        exp_num=$((exp_num+1))
      done
    done
  done
done

echo "Finished running $exp_num experiments"
