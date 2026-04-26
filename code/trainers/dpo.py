import os
os.environ["WANDB__SERVICE_WAIT"] = "10000"
os.environ["WANDB_INIT_TIMEOUT"] = "10000"
os.environ['WANDB_START_METHOD'] = 'thread'

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore")

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trainers.network_utils import AutoModelForCausalLMWithValueHead
from trainers.dpo_trainer import DPOTrainer
from trainers.dpo_config import DPOConfig
import torch
from absl import flags, app
import accelerate
import gc
import datetime
import numpy as np
import tempfile
from tqdm import tqdm
import wandb  # noqa: F401
from collections import defaultdict
import re

from trainers.utils import (
    logprobs_from_logits,
    entropy_from_logits,
)

FLAGS = flags.FLAGS
flags.DEFINE_string('wandb_project', 'reweighted_bc', 'the wandb project name')
flags.DEFINE_string('run_name', 'reweighted_bc', 'the wandb run name')
flags.DEFINE_string('output_dir', None, 'the output directory')
flags.DEFINE_string('dataset_path', "tatsu-lab/alpaca_farm", 'the optional supervised dataset')
flags.DEFINE_string('tokenizer_type', "EleutherAI/pythia-1.4b-deduped", 'tokenizer/model name')
flags.DEFINE_string('pretrained_dir', "", 'path or HF id to pretrained model')
flags.DEFINE_string('ref_model_path', None, 'path to reference model (defaults to pretrained_dir)')

flags.DEFINE_float('learning_rate', 1.0e-6, 'learning rate')
flags.DEFINE_integer('num_train_epochs', 50, 'training epochs')
flags.DEFINE_integer('inner_iteration_steps', 1, 'inner optimization epochs')
flags.DEFINE_integer('batch_size', 64, 'batch size')
flags.DEFINE_integer('mini_batch_size', 8, 'mini-batch size')
flags.DEFINE_integer('seed', 42, 'random seed')
flags.DEFINE_integer('gradient_accumulation_steps', 1, 'gradient accumulation steps')
flags.DEFINE_float('beta', 0.05, 'DPO β')

# preference dataset
flags.DEFINE_string('preference_dataset_path', 'Asap7772/relabeled_alpacafarm_pythiasft_20K_preference_data_minlength', 'preference dataset path')
flags.DEFINE_integer('preference_num_samples', -1, 'number of samples to use (-1 = use all)')
flags.DEFINE_integer('num_proc', 32, 'parallel preprocessing')
flags.DEFINE_string('cache_dir', '', 'cache directory')
flags.DEFINE_bool('bf16', False, 'Use bfloat16')
flags.DEFINE_bool('fp16', False, 'Use float16')
flags.DEFINE_integer('max_prompt_tokens', 512, 'maximum prompt tokens')
flags.DEFINE_integer('max_response_tokens', 512, 'maximum response tokens')

flags.DEFINE_bool('xpo', False, 'Use χPO (chi-squared Preference Optimization)')
flags.DEFINE_float('xpo_r_max', 1.0, 'R_max for χPO reward clipping range [-2R_max, 2R_max]')

flags.DEFINE_bool('auxdpo', False, 'Use AuxDPO (auxiliary variable DPO)')
flags.DEFINE_float('auxdpo_lambda_null', 1.0, 'Null-space penalty weight')
flags.DEFINE_float('auxdpo_lambda_amp', 0.01, 'Delta L2 regularizer weight')
flags.DEFINE_float('auxdpo_delta_cap', 1.0, 'Delta clamp value')
flags.DEFINE_float('auxdpo_aux_lr', 5e-3, 'Learning rate for auxiliary params')

# LoRA / PEFT flags
flags.DEFINE_bool('use_lora', False, 'Use LoRA for parameter-efficient fine-tuning')
flags.DEFINE_integer('lora_r', 8, 'LoRA rank')
flags.DEFINE_integer('lora_alpha', 16, 'LoRA alpha scaling factor')
flags.DEFINE_float('lora_dropout', 0.05, 'LoRA dropout probability')
flags.DEFINE_string('lora_target_modules', 'q_proj,k_proj,v_proj,o_proj',
                    'Comma-separated list of modules to apply LoRA to')

PROMPT_TOKEN = '<|prompter|>'     # overridden after tokenizer loads
ASSISTANT_TOKEN = '<|assistant|>'
EOS_TOKEN = '<|endoftext|>'


def main(_):
    rng = np.random.RandomState(FLAGS.seed)
    output_dir = f"{FLAGS.output_dir}/{FLAGS.wandb_project}/{FLAGS.run_name}"
    model_name = f"{FLAGS.wandb_project}_{FLAGS.run_name}"

    # ==== Load preference dataset ====
    raw = load_dataset(FLAGS.preference_dataset_path)

    def _has_preference_columns(ds):
        cols = ds[list(ds.keys())[0]].column_names
        return 'prompt' in cols and 'y_w' in cols and 'y_l' in cols

    def _convert_mmlu_pro(dataset, seed):
        """Convert MMLU-PRO (question/options/answer) to preference pairs."""
        _rng = np.random.RandomState(seed)
        def _to_pref(batch):
            new = defaultdict(list)
            for q, opts, ans_idx in zip(batch['question'], batch['options'], batch['answer_index']):
                # Build prompt with options listed
                opt_str = "\n".join(f"({chr(65+i)}) {o}" for i, o in enumerate(opts))
                prompt = f"{q}\n\n{opt_str}"
                chosen = opts[ans_idx]
                # Pick a random wrong answer
                wrong_indices = [i for i in range(len(opts)) if i != ans_idx]
                rejected = opts[_rng.choice(wrong_indices)]
                new['prompt'].append(prompt)
                new['y_w'].append(f"({chr(65+ans_idx)}) {chosen}")
                new['y_l'].append(f"({chr(65+wrong_indices[0])}) {rejected}")
            return new
        return dataset.map(_to_pref, batched=True, num_proc=FLAGS.num_proc,
                           remove_columns=dataset.column_names)

    if _has_preference_columns(raw):
        # Dataset already has prompt/y_w/y_l (e.g. alpaca_farm)
        split_eval = 'eval' if 'eval' in raw else ('validation' if 'validation' in raw else list(raw.keys())[-1])
        pref_dataset, eval_pref_dataset = raw['train'], raw[split_eval]
    elif 'question' in raw[list(raw.keys())[0]].column_names and 'options' in raw[list(raw.keys())[0]].column_names:
        # MMLU-PRO format: no train split, use test as train, validation as eval
        train_split = 'train' if 'train' in raw else 'test'
        eval_split = 'validation' if 'validation' in raw else list(raw.keys())[-1]
        pref_dataset = _convert_mmlu_pro(raw[train_split], FLAGS.seed)
        eval_pref_dataset = _convert_mmlu_pro(raw[eval_split], FLAGS.seed)
    else:
        raise ValueError(f"Unknown dataset format. Columns: {raw[list(raw.keys())[0]].column_names}")

    def process_dataset_initial(batch):
        new = defaultdict(list)
        for p, ch, rj in zip(batch['prompt'], batch['y_w'], batch['y_l']):
            text = f"{PROMPT_TOKEN}{p}{ASSISTANT_TOKEN}"
            new['prompt'].append(text)
            new['response_w'].append(ch)
            new['response_l'].append(rj)
        return new

    pref_dataset = pref_dataset.map(process_dataset_initial, batched=True, num_proc=FLAGS.num_proc)
    eval_pref_dataset = eval_pref_dataset.map(process_dataset_initial, batched=True, num_proc=FLAGS.num_proc)

    pref_dataset = pref_dataset.shuffle(seed=FLAGS.seed)
    if FLAGS.preference_num_samples > 0:
        pref_dataset = pref_dataset.select(range(min(FLAGS.preference_num_samples, len(pref_dataset))))
    eval_pref_dataset = eval_pref_dataset.shuffle(seed=FLAGS.seed).select(range(len(eval_pref_dataset)))

    # Add persistent example indices for AuxDPO δ lookup
    pref_dataset = pref_dataset.map(lambda ex, idx: {"example_idx": idx}, with_indices=True)

    print(f"[DPO] train={len(pref_dataset)} eval={len(eval_pref_dataset)}")

    # def process_dataset(batch):
    #     new = {}
    #     new['query'] = batch['prompt']
    #     new['text_w'] = batch['y_w']
    #     new['text_l'] = batch['y_l']
    #     new['response_w'] = [x.split(ASSISTANT_TOKEN)[-1] for x in batch['y_w']]
    #     new['response_l'] = [x.split(ASSISTANT_TOKEN)[-1] for x in batch['y_l']]
    #     return new

    # pref_dataset = pref_dataset.map(process_dataset, batched=True, num_proc=FLAGS.num_proc,
    #                                 remove_columns=['prompt', 'chosen', 'rejected', 'y_w', 'y_l'])
    # eval_pref_dataset = eval_pref_dataset.map(process_dataset, batched=True, num_proc=FLAGS.num_proc,
    #                                           remove_columns=['prompt', 'chosen', 'rejected', 'y_w', 'y_l'])

    # ==== Config ====
    unique_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S.%f") + '-' + str(np.random.randint(100000))
    wandb_output_dir = tempfile.mkdtemp()
    config = DPOConfig(
        model_name=FLAGS.pretrained_dir,
        gradient_accumulation_steps=FLAGS.gradient_accumulation_steps,
        learning_rate=FLAGS.learning_rate,
        batch_size=FLAGS.batch_size,
        mini_batch_size=FLAGS.mini_batch_size,
        ppo_epochs=FLAGS.inner_iteration_steps,
        tracker_project_name=FLAGS.wandb_project,
        temperature=FLAGS.beta,
        log_with='wandb',
        seed=FLAGS.seed,
        project_kwargs={'project_dir': output_dir},
        tracker_kwargs={"wandb": {"name": FLAGS.run_name, "id": unique_str, "dir": wandb_output_dir}},
        xpo=FLAGS.xpo,
        xpo_r_max=FLAGS.xpo_r_max,
        auxdpo=FLAGS.auxdpo,
        auxdpo_lambda_null=FLAGS.auxdpo_lambda_null,
        auxdpo_lambda_amp=FLAGS.auxdpo_lambda_amp,
        auxdpo_delta_cap=FLAGS.auxdpo_delta_cap,
        auxdpo_aux_lr=FLAGS.auxdpo_aux_lr,
        use_lora=FLAGS.use_lora,
        lora_r=FLAGS.lora_r,
        lora_alpha=FLAGS.lora_alpha,
        lora_dropout=FLAGS.lora_dropout,
        lora_target_modules=FLAGS.lora_target_modules,
    )

    # ==== Tokenizer & models ====
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.tokenizer_type)
    if tokenizer.pad_token is None:
        # Use an existing special token as pad to avoid adding a new untrained embedding
        if 'llama' in FLAGS.tokenizer_type.lower() or 'Llama' in FLAGS.tokenizer_type:
            # LLaMA 3.x has a dedicated pad token already in vocab
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|padding|>"})
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    # Set special tokens based on tokenizer
    global PROMPT_TOKEN, ASSISTANT_TOKEN, EOS_TOKEN
    eos = tokenizer.eos_token or '<|endoftext|>'
    EOS_TOKEN = eos
    # Since prompt and response are tokenized separately, we only need
    # simple formatting — no special separator tokens needed
    PROMPT_TOKEN = 'Question: '
    ASSISTANT_TOKEN = '\nAnswer: '
    dtype = torch.float32
    if FLAGS.bf16: dtype = torch.bfloat16
    elif FLAGS.fp16: dtype = torch.float16

    policy = AutoModelForCausalLM.from_pretrained(
        FLAGS.pretrained_dir, cache_dir=FLAGS.cache_dir,
        torch_dtype=dtype, low_cpu_mem_usage=True,
        device_map='auto', trust_remote_code=True)
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Apply LoRA if requested
    if FLAGS.use_lora:
        from peft import LoraConfig, get_peft_model
        target_modules = [m.strip() for m in FLAGS.lora_target_modules.split(',')]
        lora_config = LoraConfig(
            r=FLAGS.lora_r,
            lora_alpha=FLAGS.lora_alpha,
            lora_dropout=FLAGS.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        policy = get_peft_model(policy, lora_config)
        policy.print_trainable_parameters()

    model = AutoModelForCausalLMWithValueHead(policy)

    ref_path = FLAGS.ref_model_path or FLAGS.pretrained_dir
    ref_policy = AutoModelForCausalLM.from_pretrained(
        ref_path, cache_dir=FLAGS.cache_dir, torch_dtype=dtype,
        low_cpu_mem_usage=True, device_map='auto', trust_remote_code=True)
    ref_model = AutoModelForCausalLMWithValueHead(ref_policy)
    for p in ref_model.parameters(): p.requires_grad = False
    ref_model.eval()

    dataset = Dataset.from_dict({"input_ids": [[tokenizer.eos_token_id]], "attention_mask": [[1]]})
    trainer = DPOTrainer(model=model, ref_model=ref_model, config=config,
                         dataset=dataset, tokenizer=tokenizer,
                         additional_config_kwargs=FLAGS.flag_values_dict())

    # Initialize persistent δ buffer for AuxDPO
    if FLAGS.auxdpo:
        trainer.init_auxdpo_deltas(len(pref_dataset))
    
    # ==== Dataloaders ====
    common_dl_kwargs = dict(shuffle=True, drop_last=True, num_workers=min(8, os.cpu_count() or 1),
                            pin_memory=True, persistent_workers=False)
    def collate_text(batch):
        keys = batch[0].keys()
        return {k: [example[k] for example in batch] for k in keys}
    pref_loader = torch.utils.data.DataLoader(
        pref_dataset,
        batch_size=FLAGS.batch_size,
        collate_fn=collate_text,
        **common_dl_kwargs,
    )

    total_len = len(pref_loader)

    MAX_PROMPT_TOKENS = FLAGS.max_prompt_tokens
    MAX_RESP_TOKENS = FLAGS.max_response_tokens

    @torch.no_grad()
    def process_pref_batch(batch):
        # Tokenize prompt and responses separately, then let _step concatenate
        pref_prompt = tokenizer(batch["prompt"], padding=True, truncation=True,
                               max_length=MAX_PROMPT_TOKENS, return_tensors='pt').input_ids
        pref_prompt = accelerate.utils.send_to_device(pref_prompt, trainer.accelerator.device)
        all_resp = batch["response_w"] + batch["response_l"]
        # Right-pad responses so cat(prompt, response) = [prompt | response | pad...]
        tokenizer.padding_side = "right"
        resp_tok = tokenizer(all_resp, padding=True, truncation=True,
                             max_length=MAX_RESP_TOKENS, return_tensors='pt',
                             add_special_tokens=False).input_ids
        tokenizer.padding_side = "left"  # restore for next prompt tokenization
        resp_w = accelerate.utils.send_to_device(resp_tok[:len(batch["response_w"])], trainer.accelerator.device)
        resp_l = accelerate.utils.send_to_device(resp_tok[len(batch["response_w"]):], trainer.accelerator.device)
        example_indices = torch.tensor(batch["example_idx"], dtype=torch.long,
                                       device=trainer.accelerator.device)
        return batch, pref_prompt, resp_w, resp_l, example_indices

    # ==== Training ====
    total_iterations = 0
    columns_to_log = ["prompt", "response_w", "response_l"]
    print("Starting training...")

    for epoch in tqdm(range(FLAGS.num_train_epochs), desc="Epochs"):
        for sub_it, batch in tqdm(enumerate(pref_loader), desc="Batches", total=total_len):
            gc.collect(); torch.cuda.empty_cache()
            batch, q_t, rw_t, rl_t, idx_t = process_pref_batch(batch)
            out_batch = {k: batch[k] for k in columns_to_log if k in batch}
            stats = trainer.step(queries=q_t, responses_w=rw_t, responses_l=rl_t, example_indices=idx_t)
            bs = q_t.shape[0]
            rewards = stats["policy/reward_margin_dist"] if "policy/reward_margin_dist" in stats else \
                      torch.zeros(bs, device=trainer.current_device)
            if not isinstance(rewards, torch.Tensor):
                rewards = torch.tensor(rewards, dtype=torch.float32, device=trainer.current_device)
            trainer.log_stats(stats=stats, batch=out_batch, rewards=rewards, columns_to_log=columns_to_log)
            total_iterations += 1

    print("Training complete.")

    # ==== Post-training evaluation on eval set ====
    print(f"[DPO] Running post-training evaluation on {len(eval_pref_dataset)} examples...")
    eval_loader = torch.utils.data.DataLoader(
        eval_pref_dataset,
        batch_size=FLAGS.batch_size,
        collate_fn=collate_text,
        shuffle=False,
        drop_last=False,
        num_workers=min(8, os.cpu_count() or 1),
        pin_memory=True,
    )
    eval_metrics = trainer.evaluate(
        eval_loader, tokenizer,
        max_prompt_tokens=MAX_PROMPT_TOKENS,
        max_response_tokens=MAX_RESP_TOKENS,
    )
    print("[DPO] Evaluation results:")
    for k, v in eval_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Log eval metrics to wandb
    if trainer.accelerator.is_main_process:
        import wandb
        wandb.log(eval_metrics)
        wandb.summary.update(eval_metrics)

    print("Evaluation complete.")

if __name__ == "__main__":
    app.run(main)
