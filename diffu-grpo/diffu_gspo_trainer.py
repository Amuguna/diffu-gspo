# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from diffu_grpo_trainer import DiffuGRPOTrainer


# class DiffuGSPOTrainer(DiffuGRPOTrainer):
#     """Sequence‑level PPO‑style trainer for diffusion language models (GSPO).

#     * Aligns reward granularity (sequence) and optimisation granularity.
#     * Implements importance‑sampling ratio, clip objective and KL penalty all at
#       the **sequence** level.
#     """

#     # ------------------------------------------------------------------
#     # Utility
#     # ------------------------------------------------------------------
#     @staticmethod
#     def _sequence_logps(per_token_logps: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
#         """Length‑normalised log‑probability of each completion sequence.

#         Args:
#             per_token_logps: `Tensor[B, T]`  – log‑probs for *completion* tokens.
#             mask           : `Tensor[B, T]`  – 1 before first EOS, 0 afterwards.
#         Returns:
#             `Tensor[B]` – log‑probability divided by effective length.
#         """
#         lengths = mask.sum(dim=1).clamp(min=1)  # avoid /0
#         return (per_token_logps * mask).sum(dim=1) / lengths

#     # ------------------------------------------------------------------
#     # Loss
#     # ------------------------------------------------------------------
#     def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch=None):
#         if return_outputs:
#             raise ValueError("DiffuGSPOTrainer does not support return_outputs=True")

#         # -------- unpack batch --------
#         prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
#         completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
#         mask_seeds = inputs["mask_seeds"]

#         input_ids = torch.cat([prompt_ids, completion_ids], dim=1)          # [B, L]
#         logits_to_keep = completion_ids.size(1)
#         itr_idx = self._step % self.args.num_iterations

#         # -------- current policy log‑probs (per‑token) --------
#         per_token_logps = self._get_per_token_logps(
#             model,
#             input_ids.unsqueeze(0),               # [1, B, L]
#             logits_to_keep,
#             [mask_seeds[itr_idx]],
#         ).squeeze(0)                             # -> [B, T]

#         seq_logps = self._sequence_logps(per_token_logps, completion_mask)  # [B]

#         # -------- old policy log‑probs --------
#         if "old_per_token_logps" in inputs:                     # preferred path
#             old_per_token_logps = inputs["old_per_token_logps"][itr_idx]
#             old_seq_logps = self._sequence_logps(old_per_token_logps, completion_mask).detach()
#         else:                                                    # fallback if only seq stored
#             old_seq_logps = inputs["old_seq_logps"][itr_idx].detach()

#         # -------- IS ratio + clip objective --------
#         ratio = torch.exp(seq_logps - old_seq_logps)            # [B]
#         ratio = torch.clamp(ratio, 0, 1e4)                      # hard clip to avoid inf

#         advantages = inputs["advantages"].squeeze()             # [B]
#         unclipped = ratio * advantages
#         clipped = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
#         policy_loss = -torch.min(unclipped, clipped).mean()

#         # -------- reverse‑KL penalty (optional) --------
#         loss = policy_loss
#         if self.beta != 0.0:
#             if "ref_per_token_logps" in inputs:
#                 ref_per_token_logps = inputs["ref_per_token_logps"][itr_idx]
#                 ref_seq_logps = self._sequence_logps(ref_per_token_logps, completion_mask)
#             else:
#                 ref_seq_logps = inputs["ref_seq_logps"][itr_idx]
#             delta = seq_logps - ref_seq_logps
#             kl_div = (torch.exp(delta) - delta - 1.0).mean()
#             loss = loss + self.beta * kl_div
#         else:
#             kl_div = torch.tensor(0.0, device=loss.device)

#         # ------------------------------------------------------------------
#         # Metrics (safe gather across processes)
#         # ------------------------------------------------------------------
#         mode = "eval" if self.control.should_evaluate else "train"
#         with torch.no_grad():
#             clip_tensor = ((ratio < 1 - self.epsilon) | (ratio > 1 + self.epsilon)).float().mean()
#             clip_ratio_val = self.accelerator.gather_for_metrics(clip_tensor).mean().item()
#             self._metrics[mode]["clip_ratio"].append(clip_ratio_val)

#             if self.beta != 0.0:
#                 kl_val = self.accelerator.gather_for_metrics(kl_div).mean().item()
#                 self._metrics[mode]["kl"].append(kl_val)

#         return loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffu_grpo_trainer import DiffuGRPOTrainer  # 원본 파일명으로 가정
from typing import Any, Callable, Optional, Union, Sized
class DiffuGSPOTrainer(DiffuGRPOTrainer):
    """Sequence‑level PPO‑style trainer for diffusion language models (GSPO).

    * Aligns reward granularity (sequence) and optimisation granularity.
    * Implements importance‑sampling ratio, clip objective and KL penalty all at
      the **sequence** level.
    * Overrides data generation to pre-compute sequence-level log-probabilities for efficiency.
    """

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @staticmethod
    def _sequence_logps(per_token_logps: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Length‑normalised log‑probability of each completion sequence.

        Args:
            per_token_logps: `Tensor[B, T]`  – log‑probs for *completion* tokens.
            mask           : `Tensor[B, T]`  – 1 before first EOS, 0 afterwards.
        Returns:
            `Tensor[B]` – log‑probability divided by effective length.
        """
        lengths = mask.sum(dim=1).clamp(min=1)  # avoid /0
        return (per_token_logps * mask).sum(dim=1) / lengths

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("DiffuGSPOTrainer does not support return_outputs=True")

        # -------- unpack batch --------
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        mask_seeds = inputs["mask_seeds"]

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)          # [B, L]
        logits_to_keep = completion_ids.size(1)
        itr_idx = self._step % self.args.num_iterations

        # -------- current policy log‑probs (per‑token) --------
        per_token_logps = self._get_per_token_logps(
            model,
            input_ids.unsqueeze(0),               # [1, B, L]
            logits_to_keep,
            [mask_seeds[itr_idx]],
        ).squeeze(0)                             # -> [B, T]

        seq_logps = self._sequence_logps(per_token_logps, completion_mask)  # [B]

        # -------- old policy log‑probs (pre-computed) --------
        # 이제 이 값들은 _generate_and_score_completions에서 미리 계산되어 저장됩니다.
        old_seq_logps = inputs["old_seq_logps"][itr_idx].detach()

        # -------- IS ratio + clip objective --------
        ratio = torch.exp(seq_logps - old_seq_logps)            # [B]
        ratio = torch.clamp(ratio, 0, 1e4)                      # hard clip to avoid inf

        advantages = inputs["advantages"].squeeze()             # [B]
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        policy_loss = -torch.min(unclipped, clipped).mean()

        # -------- reverse‑KL penalty (optional) --------
        loss = policy_loss
        if self.beta != 0.0:
            # 이 값도 미리 계산되어 저장됩니다.
            ref_seq_logps = inputs["ref_seq_logps"][itr_idx]
            delta = seq_logps - ref_seq_logps
            kl_div = (torch.exp(delta) - delta - 1.0).mean()
            loss = loss + self.beta * kl_div
        else:
            kl_div = torch.tensor(0.0, device=loss.device)

        # ------------------------------------------------------------------
        # Metrics (safe gather across processes)
        # ------------------------------------------------------------------
        mode = "eval" if self.control.should_evaluate else "train"
        with torch.no_grad():
            clip_tensor = ((ratio < 1 - self.epsilon) | (ratio > 1 + self.epsilon)).float().mean()
            clip_ratio_val = self.accelerator.gather_for_metrics(clip_tensor).mean().item()
            self._metrics[mode]["clip_ratio"].append(clip_ratio_val)

            if self.beta != 0.0:
                kl_val = self.accelerator.gather_for_metrics(kl_div).mean().item()
                self._metrics[mode]["kl"].append(kl_val)

        return loss

    # ------------------------------------------------------------------
    # Data Generation (Overridden for Efficiency)
    # ------------------------------------------------------------------
    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """
        Overrides the parent method to pre-compute and store sequence-level log-probabilities,
        avoiding redundant calculations in `compute_loss`.
        """
        # 부모 클래스의 메서드를 호출하여 기본 로직(생성, 보상 계산, 토큰 로그 확률 계산)을 수행합니다.
        # This will return a dict containing 'old_per_token_logps' and 'ref_per_token_logps'.
        processed_inputs = super()._generate_and_score_completions(inputs)

        completion_mask = processed_inputs["completion_mask"]

        # Pre-compute and store sequence-level log probabilities for the old policy
        all_old_seq_logps = []
        if self.num_iterations > 1:
            for old_per_token_logps_iter in processed_inputs["old_per_token_logps"]:
                seq_logps = self._sequence_logps(old_per_token_logps_iter, completion_mask)
                all_old_seq_logps.append(seq_logps)
            processed_inputs["old_seq_logps"] = torch.stack(all_old_seq_logps)
        
        # 이제 per-token 데이터는 필요 없으므로 메모리에서 제거할 수 있습니다.
        if "old_per_token_logps" in processed_inputs:
            del processed_inputs["old_per_token_logps"]

        # Pre-compute and store sequence-level log probabilities for the reference policy
        all_ref_seq_logps = []
        if self.beta != 0.0:
            for ref_per_token_logps_iter in processed_inputs["ref_per_token_logps"]:
                seq_logps = self._sequence_logps(ref_per_token_logps_iter, completion_mask)
                all_ref_seq_logps.append(seq_logps)
            processed_inputs["ref_seq_logps"] = torch.stack(all_ref_seq_logps)

        if "ref_per_token_logps" in processed_inputs:
            del processed_inputs["ref_per_token_logps"]

        return processed_inputs