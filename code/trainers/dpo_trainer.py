import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers import logging
from typing import Optional, Union, Dict, List, Tuple
import numpy as np
import typing
from trainers.dpo_config import DPOConfig
from trainers.utils import (
    logprobs_from_logits,
    entropy_from_logits,
    masked_mean,
    masked_mean_sum,
    flatten_dict,
    set_seed,
    is_torch_greater_2_0,
)
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, is_deepspeed_available
import warnings
from transformers import DataCollatorForLanguageModeling
from torch.optim import Adam
import inspect
from packaging import version
import datasets
import tqdm

PreTrainedModelWrapper = typing.Union[nn.Module, nn.DataParallel]

class DPOTrainer():
    
    def __init__(
        self,
        config: DPOConfig = None,
        model: PreTrainedModelWrapper = None,
        ref_model: Optional[PreTrainedModelWrapper] = None,
        tokenizer: PreTrainedTokenizerBase = None,
        dataset: Optional[Union[torch.utils.data.Dataset, Dataset]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        data_collator: Optional[typing.Callable] = None,
        num_shared_layers: Optional[int] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        additional_config_kwargs: Optional[dict] = None,
    ):
        self.config = config

        set_seed(self.config.seed)
        
        # Initialize Accelerator
        self.accelerator = Accelerator(
            log_with=config.log_with,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            project_config=ProjectConfiguration(**config.project_kwargs),
            **config.accelerator_kwargs,
        )
        
        self.model = model
        
        # Reference model MUST be provided
        if ref_model is None:
            raise ValueError("ref_model must be explicitly provided! Do not rely on automatic creation.")
        self.ref_model = ref_model
        
        # Ensure reference model is frozen
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()

        self.model_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.is_encoder_decoder = hasattr(self.model, "is_encoder_decoder")
        if self.is_encoder_decoder: 
            raise ValueError("DPO does not support encoder-decoder models.")

        self.is_peft_model = getattr(self.model, "is_peft_model", False)
        config.is_encoder_decoder = self.is_encoder_decoder
        config.is_peft_model = self.is_peft_model

        is_using_tensorboard = config.log_with is not None and config.log_with == "tensorboard"
        current_config = dict(trl_dpo_trainer_config=config.to_dict()) if not is_using_tensorboard else config.to_dict()
        current_config.update(flatten_dict(additional_config_kwargs or {}))
        
        self.accelerator.init_trackers(
            config.tracker_project_name,
            config=current_config,
            init_kwargs=config.tracker_kwargs,
        )
        
        self.tokenizer = tokenizer
        self.dataset = dataset
        self._signature_columns = None
        
        if self.dataset is not None:
            self.dataloader = self.prepare_dataloader(self.dataset, data_collator)
        else:
            self.dataloader = None

        # Initialize optimizer
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        if optimizer is None:
            self.optimizer = Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config.learning_rate,
            )
        else:
            self.optimizer = optimizer

        self.lr_scheduler = lr_scheduler

        # Prepare with accelerator
        (
            self.model,
            self.optimizer,
            self.data_collator,
            self.dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.data_collator,
            self.dataloader,
            self.lr_scheduler,
        )
        
        # Prepare ref_model (but don't include in optimizer!)
        is_deepspeed_used = self.accelerator.distributed_type == "DEEPSPEED"
        if is_deepspeed_used:
            if not self.is_peft_model:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
        else:
            self.ref_model = self.accelerator.prepare(self.ref_model)

        self.is_distributed = self.accelerator.distributed_type == "MULTI_GPU"
        self.current_step = 0
        self.current_device = self.accelerator.device



        # === AuxDPO: per-example auxiliary offsets ===
        self.auxdpo = getattr(config, "auxdpo", False)
        if self.auxdpo:
            self.auxdpo_lambda_null = config.auxdpo_lambda_null
            self.auxdpo_lambda_amp = config.auxdpo_lambda_amp
            self.auxdpo_delta_cap = config.auxdpo_delta_cap
            self.auxdpo_aux_lr = config.auxdpo_aux_lr

    def init_auxdpo_deltas(self, n_examples):
        """
        Initialize persistent δ buffer of size 2N registered on the model,
        and add it to the main optimizer as a separate param group.
        """
        # Access the unwrapped model (past Accelerate/DDP wrappers)
        unwrapped = self.accelerator.unwrap_model(self.model)
        delta_raw = nn.Parameter(
            torch.zeros(2 * n_examples, dtype=torch.float32, device=self.current_device)
        )
        nn.init.normal_(delta_raw, mean=0.0, std=1e-3)
        unwrapped.register_parameter("aux_delta_raw", delta_raw)
        self.delta_raw = delta_raw

        # Add δ to main optimizer as its own param group with separate lr
        self.optimizer.add_param_group({
            "params": [self.delta_raw],
            "lr": self.auxdpo_aux_lr,
            "weight_decay": 0.0,
        })
        
    def prepare_dataloader(self, dataset: Union[torch.utils.data.Dataset, Dataset], data_collator=None):
        if isinstance(dataset, Dataset):
            dataset = self._remove_unused_columns(dataset)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.dataloader_batch_size or self.config.batch_size,
            collate_fn=data_collator,
            shuffle=True,
            drop_last=True,
        )
        return dataloader
    
    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            self._signature_columns += ["label", "query", "response"]

    def _remove_unused_columns(self, dataset: "Dataset"):
        if not self.config.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns
        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        columns = [k for k in signature_columns if k in dataset.column_names]

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"],
                columns=columns,
                format_kwargs=dataset.format["format_kwargs"],
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)

    def _step(self, queries, responses_w, responses_l, return_stats=False, preference_mask=None, delta_w=None, delta_l=None):
        # queries = prompt tokens, responses_w/l = response-only tokens
        input_ids_w = torch.cat((queries, responses_w), dim=1)
        input_ids_l = torch.cat((queries, responses_l), dim=1)

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask_w = (input_ids_w != self.tokenizer.pad_token_id).long()
        attention_mask_l = (input_ids_l != self.tokenizer.pad_token_id).long()

        # Mask out query tokens, keep only response tokens, remove last token
        query_length = queries.shape[1]
        mask_w = attention_mask_w.clone()
        mask_w[:, :query_length] = 0
        mask_w = mask_w[:, :-1]

        mask_l = attention_mask_l.clone()
        mask_l[:, :query_length] = 0
        mask_l = mask_l[:, :-1]
        
        def process_input_ids(input_ids, attention_mask):
            input_data = {"input_ids": input_ids, "attention_mask": attention_mask}
            logits, _, _ = self.model(**input_data)
            
            with torch.no_grad():
                old_logits, _, _ = self.ref_model(**input_data)
                old_logprobs = logprobs_from_logits(old_logits[:, :-1, :], input_ids[:, 1:])

            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            entropy = entropy_from_logits(logits)
            return logprobs, old_logprobs, entropy, logits
    
        logprobs_w, old_logprobs_w, entropy_w, logits_w = process_input_ids(input_ids_w, attention_mask_w)
        logprobs_l, old_logprobs_l, entropy_l, logits_l = process_input_ids(input_ids_l, attention_mask_l)

        # Debug: detect NaN source
        if torch.isnan(logits_w).any():
            nan_in_weights = any(torch.isnan(p).any() for p in self.model.parameters())
            print(f"\n[NaN DEBUG] logits_w has NaN. Model weights have NaN: {nan_in_weights}")
            if not nan_in_weights:
                print(f"  input_ids_w shape: {input_ids_w.shape}")
                print(f"  input_ids_w max: {input_ids_w.max().item()}, min: {input_ids_w.min().item()}")
                print(f"  logits_w max finite: {logits_w[torch.isfinite(logits_w)].max().item() if torch.isfinite(logits_w).any() else 'none'}")
                print(f"  logits_w NaN count: {torch.isnan(logits_w).sum().item()} / {logits_w.numel()}")
            raise RuntimeError("NaN detected — stopping to preserve debug info")
        
        # Sequence-level DPO: sum log-probs over response tokens per example
        mask_w_float = mask_w.to(dtype=logprobs_w.dtype)
        mask_l_float = mask_l.to(dtype=logprobs_l.dtype)

        # π log-ratios: mean(log π_w) - mean(log π_l) per example → (batch_size,)
        # Use mean to keep magnitudes stable regardless of sequence length
        len_w = mask_w_float.sum(dim=1).clamp(min=1)
        len_l = mask_l_float.sum(dim=1).clamp(min=1)
        pi_logp_w_seq = (logprobs_w * mask_w_float).sum(dim=1) / len_w
        pi_logp_l_seq = (logprobs_l * mask_l_float).sum(dim=1) / len_l
        ref_logp_w_seq = (old_logprobs_w * mask_w_float).sum(dim=1) / len_w
        ref_logp_l_seq = (old_logprobs_l * mask_l_float).sum(dim=1) / len_l

        pi_logratios_seq = pi_logp_w_seq - pi_logp_l_seq
        ref_logratios_seq = ref_logp_w_seq - ref_logp_l_seq

        # Per-side log-ratios log(π/π_ref): needed for χPO's non-linear link
        logratio_w = pi_logp_w_seq - ref_logp_w_seq
        logratio_l = pi_logp_l_seq - ref_logp_l_seq

        if getattr(self.config, "xpo", False):
            # χPO (Huang et al., 2024): replace DPO's log(z) link with ϕ(z) = z + log(z),
            # then clip β·ϕ(·) to [-2R_max, +2R_max] per side before forming the margin.
            # Using a numerically-stable form: β·ϕ(z) = exp(clip(log β + log z, -88, 20)) + β·log z
            # (since β·z = exp(log β + log z)). For β<0, fall back to the direct formula.
            beta = self.config.temperature
            r_max = getattr(self.config, "xpo_r_max", 1.0)
            clip_val = 2.0 * r_max

            if beta > 0:
                log_beta = math.log(beta)
                beta_ratio_w = torch.exp(torch.clamp(log_beta + logratio_w, min=-88.0, max=20.0))
                beta_ratio_l = torch.exp(torch.clamp(log_beta + logratio_l, min=-88.0, max=20.0))
            else:
                beta_ratio_w = beta * torch.exp(logratio_w)
                beta_ratio_l = beta * torch.exp(logratio_l)

            beta_phi_w = torch.clamp(beta_ratio_w + beta * logratio_w, min=-clip_val, max=clip_val)
            beta_phi_l = torch.clamp(beta_ratio_l + beta * logratio_l, min=-clip_val, max=clip_val)

            if delta_w is not None and delta_l is not None:
                aux_margin = delta_w - delta_l
                dpo_logit_seq = (beta_phi_w - beta_phi_l) + aux_margin
            else:
                dpo_logit_seq = beta_phi_w - beta_phi_l

            dpo_loss = -F.logsigmoid(dpo_logit_seq).mean()
        else:
            # === AuxDPO: augment the margin with δ (outside β) ===
            if delta_w is not None and delta_l is not None:
                aux_margin = delta_w - delta_l  # per-example scalars, shape (batch_size,)
                dpo_logit_seq = self.config.temperature * (pi_logratios_seq - ref_logratios_seq) + aux_margin
            else:
                dpo_logit_seq = self.config.temperature * (pi_logratios_seq - ref_logratios_seq)

            if getattr(self.config, "ipo_loss", False):
                dpo_loss = ((dpo_logit_seq - 1/(2 * self.config.temperature))**2).mean()
            else:
                dpo_loss = -F.logsigmoid(dpo_logit_seq).mean()

        logprobs = torch.cat((logprobs_w, logprobs_l), dim=0)
        old_logprobs = torch.cat((old_logprobs_w, old_logprobs_l), dim=0)
        cat_mask = torch.cat((mask_w, mask_l), dim=0)

        rewards = self.config.temperature * ((logprobs - old_logprobs) * cat_mask).detach()
        rewards_chosen = self.config.temperature * ((logprobs_w - old_logprobs_w) * mask_w_float).detach()
        rewards_rejected = self.config.temperature * ((logprobs_l - old_logprobs_l) * mask_l_float).detach()
        reward_margin = rewards_chosen - rewards_rejected

        logits = torch.cat((logits_w, logits_l), dim=0)
        entropy = entropy_from_logits(logits)
        approxkl = 0.5 * masked_mean((logprobs - old_logprobs) ** 2, cat_mask)
        policykl = masked_mean(logprobs - old_logprobs, cat_mask)

        sequence_approxkl = 0.5 * masked_mean_sum((logprobs - old_logprobs) ** 2, cat_mask)
        sequence_policykl = masked_mean_sum(logprobs - old_logprobs, cat_mask)

        # Sequence-level per-example rewards
        rewards_chosen_seq = (rewards_chosen * mask_w_float).sum(dim=1)
        rewards_rejected_seq = (rewards_rejected * mask_l_float).sum(dim=1)
        reward_margin_per_example = rewards_chosen_seq - rewards_rejected_seq

        if return_stats:
            # Sequence-level log-probs for logging (mean over response tokens)
            logprobs_w_seq = (logprobs_w * mask_w_float).sum(dim=1) / len_w
            logprobs_l_seq = (logprobs_l * mask_l_float).sum(dim=1) / len_l
            old_logprobs_w_seq = (old_logprobs_w * mask_w_float).sum(dim=1) / len_w
            old_logprobs_l_seq = (old_logprobs_l * mask_l_float).sum(dim=1) / len_l

            stats = dict(
                loss=dict(
                    dpo_loss=dpo_loss.detach(),
                ),
                policy=dict(
                    entropy=entropy.detach(),
                    entropy_w=entropy_w.detach(),
                    entropy_l=entropy_l.detach(),
                    approxkl=approxkl.detach(),
                    policykl=policykl.detach(),
                    sequence_approxkl=sequence_approxkl.detach(),
                    sequence_policykl=sequence_policykl.detach(),
                    rewards_mean=torch.mean(rewards).detach(),
                    rewards_std=torch.std(rewards).detach(),
                    rewards_dist=rewards.detach(),
                    rewards_chosen=torch.mean(rewards_chosen).detach(),
                    rewards_rejected=torch.mean(rewards_rejected).detach(),
                    reward_margin=torch.mean(reward_margin).detach(),
                    rewards_chosen_dist=rewards_chosen_seq.detach().cpu(),
                    rewards_rejected_dist=rewards_rejected_seq.detach().cpu(),
                    reward_margin_dist=reward_margin_per_example.detach().cpu(),
                    logprobs_w=torch.mean(logprobs_w_seq).detach(),
                    logprobs_l=torch.mean(logprobs_l_seq).detach(),
                    old_logprobs_w=torch.mean(old_logprobs_w_seq).detach(),
                    old_logprobs_l=torch.mean(old_logprobs_l_seq).detach(),
                    pi_logratios=torch.mean(pi_logratios_seq).detach(),
                    ref_logratios=torch.mean(ref_logratios_seq).detach(),
                    dpo_logit_mean=torch.mean(dpo_logit_seq).detach(),
                    dpo_logit_std=torch.std(dpo_logit_seq).detach(),
                    dpo_logit_dist=dpo_logit_seq.detach(),
                    classifier_accuracy=torch.mean((reward_margin_per_example > 0).float()).detach(),
                    ref_classifier_accuracy=torch.mean((old_logprobs_w_seq > old_logprobs_l_seq).float()).detach()
                )
            )


            return dpo_loss, flatten_dict(stats)
        else:
            return dpo_loss
    
    def step(self, queries, responses_w, responses_l, preference_mask=None, example_indices=None):
        assert queries.ndim == 2 and responses_w.ndim == 2 and responses_l.ndim == 2
        self.model.train()
        self.ref_model.eval()

        bs = self.config.batch_size
        sub_bs = self.config.mini_batch_size
        assert bs % sub_bs == 0

        stats = None
        for i in range(0, bs, sub_bs):
            queries_ = queries[i : i + sub_bs]
            responses_w_ = responses_w[i : i + sub_bs]
            responses_l_ = responses_l[i : i + sub_bs]
            preference_mask_ = preference_mask[i : i + sub_bs] if preference_mask is not None else None

            if self.auxdpo:
                indices_ = example_indices[i : i + sub_bs]

                # Look up persistent δ: interleaved layout [δ_w_0, δ_l_0, δ_w_1, δ_l_1, ...]
                delta_vec = self.auxdpo_delta_cap * torch.tanh(self.delta_raw)
                capped_dw = delta_vec[2 * indices_]
                capped_dl = delta_vec[2 * indices_ + 1]

                loss, stats = self._step(
                    queries=queries_, responses_w=responses_w_, responses_l=responses_l_,
                    return_stats=True, preference_mask=preference_mask_,
                    delta_w=capped_dw, delta_l=capped_dl,
                )

                # Null-space penalty
                input_ids_w = torch.cat((queries_, responses_w_), dim=1)
                input_ids_l = torch.cat((queries_, responses_l_), dim=1)
                attention_mask_w = (input_ids_w != self.tokenizer.pad_token_id).long()
                attention_mask_l = (input_ids_l != self.tokenizer.pad_token_id).long()

                null_penalty = self._auxdpo_null_penalty(
                    capped_dw, capped_dl,
                    input_ids_w, input_ids_l,
                    attention_mask_w, attention_mask_l,
                    query_length=queries_.shape[1],
                )
                # Anti-collapse: negative sign encourages δ away from trivial zero solution
                amp_penalty = -(capped_dw.pow(2).mean() + capped_dl.pow(2).mean()) * 0.5

                total_loss = loss + self.auxdpo_lambda_null * null_penalty + self.auxdpo_lambda_amp * amp_penalty

                stats["auxdpo/null_penalty"] = null_penalty.detach()
                stats["auxdpo/amp_penalty"] = amp_penalty.detach()
                stats["auxdpo/delta_w_mean"] = capped_dw.detach().mean()
                stats["auxdpo/delta_l_mean"] = capped_dl.detach().mean()
                stats["auxdpo/delta_mean_abs"] = torch.cat([capped_dw, capped_dl]).abs().mean().detach()
                stats["auxdpo/total_loss"] = total_loss.detach()

                # Joint update — single optimizer handles both θ and δ param groups
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"[WARNING] NaN/Inf loss at step {self.current_step}, skipping batch")
                    self.current_step += 1
                    continue
                self.optimizer.zero_grad()
                self.accelerator.backward(total_loss)
                grad_nan = False
                for p in self.model.parameters():
                    if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                        grad_nan = True
                        break
                if grad_nan:
                    print(f"[WARNING] NaN/Inf in gradients at step {self.current_step}, skipping update")
                    self.optimizer.zero_grad()
                    self.current_step += 1
                    continue
                self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.current_step += 1
            else:
                # Standard DPO path
                loss, stats = self._step(
                    queries=queries_,
                    responses_w=responses_w_,
                    responses_l=responses_l_,
                    return_stats=True,
                    preference_mask=preference_mask_,
                )
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[WARNING] NaN/Inf loss at step {self.current_step}, skipping batch")
                    self.current_step += 1
                    continue
                self.optimizer.zero_grad()
                self.accelerator.backward(loss)
                # Check for NaN in gradients before stepping
                grad_nan = False
                for p in self.model.parameters():
                    if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                        grad_nan = True
                        break
                if grad_nan:
                    print(f"[WARNING] NaN/Inf in gradients at step {self.current_step}, skipping update")
                    self.optimizer.zero_grad()
                    self.current_step += 1
                    continue
                self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.current_step += 1

        return stats

    def log_stats(
            self,
            stats: dict,
            batch: dict,
            rewards: torch.FloatTensor,
            columns_to_log: List[str] = ["query", "response"],
        ):
            if self.accelerator.is_main_process:
                logs = {}

                if not isinstance(rewards, torch.Tensor):
                    rewards = torch.tensor(rewards).to(self.current_device)

                if self.config.log_with == "wandb":
                    import wandb

                    if any([column_to_log not in batch.keys() for column_to_log in columns_to_log]):
                        raise ValueError(f"Columns to log {columns_to_log} are not present in the batch {batch.keys()}.")

                    batch_list = [batch[column_to_log] for column_to_log in columns_to_log]
                    table_rows = [list(r) for r in zip(*batch_list, rewards.cpu().tolist())]
                    logs.update({"game_log": wandb.Table(columns=[*columns_to_log, "reward"], rows=table_rows)})

                logs.update(stats)

                # Cast bf16 to fp32 for logging
                for k, v in list(logs.items()):
                    if isinstance(v, torch.Tensor):
                        if v.dtype == torch.bfloat16:
                            v = v.float()
                        if torch.isnan(v).any() or torch.isinf(v).any():
                            if v.dim() == 0:
                                # Scalar NaN = real training problem, log as 0 but warn
                                print(f"[WARNING] NaN/Inf in scalar metric '{k}' at step {self.current_step}")
                            # Replace NaN/Inf with 0 so wandb histogram doesn't crash
                            v = torch.where(torch.isfinite(v), v, torch.zeros_like(v))
                        logs[k] = v
                    elif isinstance(v, np.ndarray):
                        if not np.isfinite(v).all():
                            print(f"[WARNING] NaN/Inf in array metric '{k}' at step {self.current_step}")
                            v = np.where(np.isfinite(v), v, 0.0)
                            logs[k] = v

                logs["dataset/reward_mean"] = torch.mean(rewards).cpu().numpy().item()
                logs["dataset/reward_std"] = torch.std(rewards).cpu().numpy().item()
                logs["dataset/reward_dist"] = rewards.cpu().numpy()

                self.accelerator.log(
                    logs,
                    step=self.current_step if self.config.log_with == "tensorboard" else None,
                )
    @torch.no_grad()
    def evaluate(self, eval_loader, tokenizer, max_prompt_tokens, max_response_tokens):
        """Run evaluation on a dataloader and return aggregated metrics."""
        self.model.eval()
        self.ref_model.eval()

        all_classifier_acc = []
        all_ref_classifier_acc = []
        all_logprobs_w = []
        all_logprobs_l = []
        all_old_logprobs_w = []
        all_old_logprobs_l = []
        all_reward_margin = []
        all_dpo_loss = []

        orig_pad_side = tokenizer.padding_side

        for batch in eval_loader:
            # Tokenize prompt
            tokenizer.padding_side = "left"
            tokenizer.truncation_side = "left"
            pref_prompt = tokenizer(
                batch["prompt"], padding=True, truncation=True,
                max_length=max_prompt_tokens, return_tensors="pt",
            ).input_ids
            pref_prompt = pref_prompt.to(self.current_device)

            # Tokenize responses (right-padded)
            all_resp = batch["response_w"] + batch["response_l"]
            tokenizer.padding_side = "right"
            resp_tok = tokenizer(
                all_resp, padding=True, truncation=True,
                max_length=max_response_tokens, return_tensors="pt",
                add_special_tokens=False,
            ).input_ids
            tokenizer.padding_side = orig_pad_side

            n = len(batch["response_w"])
            resp_w = resp_tok[:n].to(self.current_device)
            resp_l = resp_tok[n:].to(self.current_device)

            # Forward pass (no auxdpo deltas for eval — pure DPO metrics)
            _, stats = self._step(
                queries=pref_prompt, responses_w=resp_w, responses_l=resp_l,
                return_stats=True,
            )

            all_classifier_acc.append(stats["policy/classifier_accuracy"])
            all_logprobs_w.append(stats["policy/logprobs_w"])
            all_logprobs_l.append(stats["policy/logprobs_l"])
            all_old_logprobs_w.append(stats["policy/old_logprobs_w"])
            all_old_logprobs_l.append(stats["policy/old_logprobs_l"])
            all_reward_margin.append(stats["policy/reward_margin"])
            all_dpo_loss.append(stats["loss/dpo_loss"])
            all_ref_classifier_acc.append(stats["policy/ref_classifier_accuracy"])

        self.model.train()

        def _mean(lst):
            t = torch.stack([x if isinstance(x, torch.Tensor) else torch.tensor(x) for x in lst])
            return t.mean().item()

        return {
            "eval/classifier_accuracy": _mean(all_classifier_acc),
            "eval/ref_classifier_accuracy": _mean(all_ref_classifier_acc),
            "eval/logprobs_w": _mean(all_logprobs_w),
            "eval/logprobs_l": _mean(all_logprobs_l),
            "eval/old_logprobs_w": _mean(all_old_logprobs_w),
            "eval/old_logprobs_l": _mean(all_old_logprobs_l),
            "eval/reward_margin": _mean(all_reward_margin),
            "eval/dpo_loss": _mean(all_dpo_loss),
        }

    def _auxdpo_null_penalty(self, delta_w, delta_l, input_ids_w, input_ids_l,
                            attention_mask_w, attention_mask_l, query_length):
        """
        Compute ‖A_θ₀ · δ‖² via finite-difference approximation:
            A_θ₀ · δ ≈ [score_ref(θ₀ + ε·δ_direction) - score_ref(θ₀)] / ε
        where score = Σ_j δ_j · log π_ref(a_j|s_j).

        This avoids autograd through the ref model entirely — two no_grad
        forward passes instead of building a full computation graph.
        Gradient flows through δ_w/δ_l only (via the finite-diff quotient).
        """
        eps = 1e-4

        # Identify ref params that match trainable model params (for the null-space)
        if not hasattr(self, '_auxdpo_grad_params'):
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_ref = self.accelerator.unwrap_model(self.ref_model)
            ref_named = dict(unwrapped_ref.named_parameters())
            self._auxdpo_grad_params = []

            is_lora = getattr(self.config, "use_lora", False)

            if is_lora:
                import re
                model_names = [n for n, p in unwrapped_model.named_parameters() if p.requires_grad]
                ref_names = list(ref_named.keys())
                print(f"[AuxDPO DEBUG] Sample trainable model params: {model_names[:3]}")
                print(f"[AuxDPO DEBUG] Sample ref params: {ref_names[:3]}")

                seen_base_names = set()
                for name, p in unwrapped_model.named_parameters():
                    if not p.requires_grad or name == "aux_delta_raw":
                        continue
                    if "lora_" not in name:
                        if name in ref_named:
                            self._auxdpo_grad_params.append(ref_named[name])
                        continue

                    m = re.search(r'base_model\.model\.(.+)\.lora_[AB]\.', name)
                    if m is None:
                        m = re.search(r'(.+)\.lora_[AB]\.', name)
                    if m is None:
                        continue

                    module_path = m.group(1)
                    candidates = [
                        f"pretrained_model.model.{module_path}.weight",
                        f"model.{module_path}.weight",
                        f"pretrained_model.{module_path}.weight",
                        f"{module_path}.weight",
                    ]
                    matched = False
                    for cand in candidates:
                        if cand in ref_named and cand not in seen_base_names:
                            seen_base_names.add(cand)
                            self._auxdpo_grad_params.append(ref_named[cand])
                            matched = True
                            break
                    if not matched and module_path not in seen_base_names:
                        suffix = f"{module_path}.weight"
                        for rn in ref_names:
                            if rn.endswith(suffix) and rn not in seen_base_names:
                                seen_base_names.add(rn)
                                self._auxdpo_grad_params.append(ref_named[rn])
                                matched = True
                                break
            else:
                for name, p in unwrapped_model.named_parameters():
                    if p.requires_grad and name != "aux_delta_raw" and name in ref_named:
                        self._auxdpo_grad_params.append(ref_named[name])

            n_grad = len(self._auxdpo_grad_params)
            n_trainable = sum(1 for n, p in unwrapped_model.named_parameters()
                              if p.requires_grad and n != "aux_delta_raw")
            print(f"[AuxDPO] Null penalty (finite-diff): {n_grad} ref params matched "
                  f"({'LoRA base weights' if is_lora else 'direct'}, "
                  f"{n_trainable} trainable model params)")

        grad_params = self._auxdpo_grad_params
        if not grad_params:
            return torch.tensor(0.0, device=self.current_device)

        unwrapped_ref = self.accelerator.unwrap_model(self.ref_model)
        ref_base = getattr(unwrapped_ref, 'pretrained_model', unwrapped_ref)

        def _ref_scores(input_ids_w_, attention_mask_w_, input_ids_l_, attention_mask_l_):
            """Compute per-example ref log-prob scores under current ref params."""
            with torch.no_grad():
                out_w = ref_base(input_ids=input_ids_w_, attention_mask=attention_mask_w_)
                logits_w = out_w.logits if hasattr(out_w, 'logits') else out_w[0]
                lp_w = logprobs_from_logits(logits_w[:, :-1, :], input_ids_w_[:, 1:])
                mask_w = attention_mask_w_.clone()
                mask_w[:, :query_length] = 0
                score_w = (lp_w * mask_w[:, :-1].float()).sum(dim=1)

                out_l = ref_base(input_ids=input_ids_l_, attention_mask=attention_mask_l_)
                logits_l = out_l.logits if hasattr(out_l, 'logits') else out_l[0]
                lp_l = logprobs_from_logits(logits_l[:, :-1, :], input_ids_l_[:, 1:])
                mask_l = attention_mask_l_.clone()
                mask_l[:, :query_length] = 0
                score_l = (lp_l * mask_l[:, :-1].float()).sum(dim=1)
            return score_w, score_l

        # Baseline: score at θ₀
        score_w_0, score_l_0 = _ref_scores(input_ids_w, attention_mask_w,
                                            input_ids_l, attention_mask_l)
        # δ-weighted scalar: Σ_j (δ_w_j · score_w_j + δ_l_j · score_l_j)
        f0 = (delta_w.detach() * score_w_0).sum() + (delta_l.detach() * score_l_0).sum()

        # Perturb ref params: θ₀ + ε * (unit direction per param)
        # The "direction" is uniform across params — we perturb all matched
        # params by +ε to get the directional derivative of the δ-weighted score.
        for p in grad_params:
            p.data.add_(eps)

        score_w_eps, score_l_eps = _ref_scores(input_ids_w, attention_mask_w,
                                                input_ids_l, attention_mask_l)
        f_eps = (delta_w.detach() * score_w_eps).sum() + (delta_l.detach() * score_l_eps).sum()

        # Restore ref params
        for p in grad_params:
            p.data.add_(-eps)

        # ‖A_θ₀ · δ‖² ≈ ((f(θ₀+ε) - f(θ₀)) / ε)²
        # Gradient flows through delta_w/delta_l via the scores (which are detached,
        # but we reconstruct the penalty as a function of δ for autograd).
        # Recompute with δ attached: use the finite-diff Jacobian as a fixed coefficient.
        jvp_estimate = (f_eps - f0) / eps  # scalar, detached from δ

        # To allow gradient flow through δ, express penalty as:
        # penalty = (Σ_j δ_w_j * g_w_j + δ_l_j * g_l_j)² where g = dscore/dθ (fixed)
        # The per-example "effective gradients" are the score differences:
        g_w = (score_w_eps - score_w_0) / eps  # (batch_size,), detached
        g_l = (score_l_eps - score_l_0) / eps  # (batch_size,), detached

        # Now build penalty with δ in the graph
        At_delta = (delta_w * g_w).sum() + (delta_l * g_l).sum()
        penalty = At_delta.pow(2)

        return penalty