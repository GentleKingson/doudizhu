from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from marvel.model.base import BaseMarvelModule, ResGatedMLP, _dynamo_disable
from marvel.model.trainable import TrainableModule
from marvel.optim.common import MultiOptimizer

if TYPE_CHECKING:
    from marvel.config import OptimizerConfig
from marvel.optim.muon import Muon

from marvel.model.transformer import TransformerConfig, TransformerModel


@dataclass(frozen = True)
class BeliefTokenOutput:
    logits: torch.Tensor  # [B, S, A]
    belief_hidden: torch.Tensor  # [B, S, H]
    full_hidden: torch.Tensor  # [B, L, H]
    reward_to_go: torch.Tensor | None = None  # [B]


class BeliefDecoderConcat(nn.Module):
    """Decoder-only sequence model over [obs-prefix rows, state row, belief rows, optional RTG context]."""

    @staticmethod
    def _build_belief_row_mlp(
        *,
        hidden_size: int,
        n_layers: int,
    ) -> nn.Module:
        n_layers = int(n_layers)
        if n_layers <= 0:
            raise ValueError(
                "model_cfg.belief_layers must be positive, "
                f"got {n_layers}."
            )

        return BaseMarvelModule._build_structured_mlp(
            input_dim = hidden_size,
            hidden_dim = hidden_size,
            n_layers = n_layers,
            output_dim = hidden_size,
        )

    @staticmethod
    def _parse_architecture_version(architecture_raw: Any) -> tuple[int, int]:
        if isinstance(architecture_raw, (tuple, list)) and len(architecture_raw) == 2:
            major = int(architecture_raw[0])
            minor = int(architecture_raw[1])
            return major, minor

        architecture = str(architecture_raw).strip().lower()
        if not architecture.startswith("v"):
            raise ValueError(
                "Unsupported yomi model architecture "
                f"{architecture_raw!r}; expected version >= v1.1."
            )
        body = architecture[1:]
        if not body:
            raise ValueError(
                "Unsupported yomi model architecture "
                f"{architecture_raw!r}; expected version >= v1.1."
            )
        parts = body.split(".")
        if len(parts) == 1:
            major = int(parts[0])
            minor = 0
            return major, minor
        if len(parts) == 2:
            major = int(parts[0])
            minor = int(parts[1])
            return major, minor
        raise ValueError(
            "Unsupported yomi model architecture "
            f"{architecture_raw!r}; expected version >= v1.1."
        )

    def __init__(self, *, game: Any, model_cfg: Any) -> None:
        super().__init__()
        self.game = game
        self.model_cfg = model_cfg
        self._wd_param_names: tuple[str, ...] | None = None

        self.num_tokens = int(game.NUM_TOKENS)
        self.pad_token_id = int(game.PAD_TOKEN)
        self.max_obs_tokens = int(game.MAX_OBSERVATION_TOKENS)
        self.max_belief_rows = int(game.MAX_BELIEF_ROWS)
        self.max_belief_tokens = int(game.MAX_BELIEF_TOKENS)
        self.max_belief_candidates = int(game.MAX_BELIEF_CANDIDATES)
        self.num_channels = int(game.NUM_CHANNELS)
        architecture_raw = getattr(model_cfg, "architecture", "v1.1")
        architecture_version = self._parse_architecture_version(architecture_raw)
        if architecture_version < (1, 1):
            raise ValueError(
                "Unsupported yomi model architecture "
                f"{architecture_raw!r}; expected version >= v1.1."
            )
        use_gated_attention = True

        hidden_size = int(model_cfg.history_dim)
        # v1.3: hard-coded Transformer FFN ratio of 2.
        transformer_ffn_ratio = 2 if architecture_version >= (1, 3) else int(getattr(model_cfg, "ffn_ratio", 2))
        intermediate_size = int(hidden_size * transformer_ffn_ratio)
        max_rows = int(game.max_game_horizon) + 1 + self.max_belief_rows + 8
        self.belief_layers = int(getattr(model_cfg, "belief_layers", 2))
        if self.belief_layers <= 0:
            raise ValueError(
                "model_cfg.belief_layers must be positive, "
                f"got {self.belief_layers}."
            )

        tr_cfg = TransformerConfig(
            vocab_size = 0,
            hidden_size = hidden_size,
            intermediate_size = intermediate_size,
            num_hidden_layers = int(model_cfg.decoder_layers),
            num_attention_heads = int(model_cfg.decoder_heads),
            hidden_act = "silu",
            max_position_embeddings = max_rows,
            initializer_range = 0.02,
            rms_norm_eps = 1e-6,
            use_cache = False,
            pad_token_id = self.pad_token_id,
            use_gated_attention = use_gated_attention,
        )
        self.decoder = TransformerModel(tr_cfg)

        self.obs_embed = nn.Embedding(self.num_tokens, hidden_size, padding_idx = self.pad_token_id)
        self.belief_in_embed = nn.Embedding(self.num_tokens, hidden_size, padding_idx = self.pad_token_id)
        self.belief_out_embed = nn.Embedding(self.num_tokens, hidden_size, padding_idx = self.pad_token_id)
        if architecture_version >= (1, 2):
            # v1.3: hard-coded ResMLP FFN ratio of 1.
            resmlp_expansion = 1 if architecture_version >= (1, 3) else int(getattr(model_cfg, "ffn_ratio", 2))
            input_proj = TrainableModule._init_module(nn.Linear(hidden_size, hidden_size))
            TrainableModule._mark_no_weight_decay(input_proj)
            self.belief_out_mlp = nn.Sequential(
                input_proj,
                ResGatedMLP(
                    hidden_size,
                    n_layers = self.belief_layers,
                    expansion_ratio = resmlp_expansion,
                ),
            )
            state_proj = TrainableModule._init_module(nn.Linear(self.num_channels, hidden_size))
            TrainableModule._mark_no_weight_decay(state_proj)
            self.state_mlp = state_proj
        else:
            self.belief_out_mlp = self._build_belief_row_mlp(
                hidden_size = hidden_size,
                n_layers = self.belief_layers,
            )
            self.state_mlp = nn.Sequential(
                nn.Linear(self.num_channels, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )
        self.candidate_logit_head = BaseMarvelModule._build_output_proj(hidden_size)
        self.enable_reward_to_go_head = bool(getattr(model_cfg, "enable_reward_to_go_head", False))
        if self.enable_reward_to_go_head:
            self.reward_to_go_head: nn.Module | None = BaseMarvelModule._build_output_proj(hidden_size, std = 0.01)
            self.reward_to_go_query = nn.Parameter(torch.empty(hidden_size))
            torch.nn.init.normal_(self.reward_to_go_query, mean = 0.0, std = 0.02)
        else:
            self.reward_to_go_head = None
            self.register_parameter("reward_to_go_query", None)
        self.segment_embed = nn.Embedding(3, hidden_size)

    def _group_params_for_wd(
        self,
    ) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter], list[str]]:
        wd_params: list[torch.nn.Parameter] = []
        no_wd_params: list[torch.nn.Parameter] = []
        wd_names: list[str] = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            lower = name.lower()
            use_wd = (
                not getattr(param, "_no_weight_decay", False)
                and param.dim() >= 2
                and not name.endswith("bias")
                and "norm" not in lower
                and "embed" not in lower
            )
            (wd_params if use_wd else no_wd_params).append(param)
            if use_wd:
                wd_names.append(name)
        return wd_params, no_wd_params, wd_names

    def configure_optimizer(
        self,
        *,
        lr: float,
        optimizer: OptimizerConfig,
    ) -> torch.optim.Optimizer | MultiOptimizer:
        wd_params, no_wd_params, wd_names = self._group_params_for_wd()
        self._wd_param_names = tuple(wd_names)
        if not wd_params and not no_wd_params:
            raise RuntimeError("No trainable parameters found for optimizer construction.")

        if optimizer.use_muon and wd_params:
            muon = Muon(
                wd_params,
                lr = float(lr),
                momentum = float(optimizer.muon_beta1),
                weight_decay = float(optimizer.weight_decay),
                adamw_beta1_for_rms_match = float(optimizer.adam_beta1),
            )
            if not no_wd_params:
                return muon
            adamw = torch.optim.AdamW(
                [{"params": no_wd_params, "weight_decay": 0.0}],
                lr = float(lr),
                betas = (float(optimizer.adam_beta1), float(optimizer.adam_beta2)),
            )
            return MultiOptimizer({"muon": muon, "adamw": adamw})

        return torch.optim.AdamW(
            [
                {"params": wd_params, "weight_decay": float(optimizer.weight_decay)},
                {"params": no_wd_params, "weight_decay": 0.0},
            ],
            lr = float(lr),
            betas = (float(optimizer.adam_beta1), float(optimizer.adam_beta2)),
        )

    @property
    def weight_decay_param_names(self) -> tuple[str, ...]:
        if self._wd_param_names is None:
            raise RuntimeError("Optimizer has not been configured; call configure_optimizer() first.")
        return self._wd_param_names

    def _pool_rows(self, embed: nn.Embedding, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise RuntimeError(f"tokens must be [B, R, T], got {tuple(tokens.shape)}")
        rows = embed(tokens.to(dtype = torch.long))
        non_pad = (tokens != self.pad_token_id).unsqueeze(-1)
        denom = torch.clamp(non_pad.sum(dim = 2), min = 1)
        pooled = (rows * non_pad.to(dtype = rows.dtype)).sum(dim = 2) / denom.to(dtype = rows.dtype)
        return pooled

    def _encode_belief_input_rows(self, tokens: torch.Tensor) -> torch.Tensor:
        return self._pool_rows(self.belief_in_embed, tokens)

    @_dynamo_disable()
    def _encode_sparse_candidates_eager(
        self,
        *,
        flat_tokens: torch.Tensor,  # [BS, A, E]
        flat_mask: torch.Tensor,  # [BS, A]
    ) -> torch.Tensor:
        if flat_tokens.ndim != 3:
            raise RuntimeError(f"flat_tokens must be [BS, A, E], got {tuple(flat_tokens.shape)}")
        if flat_mask.shape != flat_tokens.shape[:2]:
            raise RuntimeError(
                "flat_mask shape mismatch: "
                f"{tuple(flat_mask.shape)} vs {tuple(flat_tokens.shape[:2])}"
            )

        batch_rows, num_candidates, _ = flat_tokens.shape
        legal_indices = flat_mask.nonzero(as_tuple = False)  # [N, 2] -> (row, candidate)
        row_idx = legal_indices[:, 0]
        cand_idx = legal_indices[:, 1]
        gathered_tokens = flat_tokens[row_idx, cand_idx]  # [N, E]
        token_embeds = self.belief_out_embed(gathered_tokens.to(dtype = torch.long)).sum(dim = -2)  # [N, H]
        encoded_sel = self.belief_out_mlp(token_embeds)
        encoded_full = token_embeds.new_zeros((batch_rows, num_candidates, token_embeds.shape[-1]))
        encoded_full[row_idx, cand_idx] = encoded_sel
        return encoded_full

    @_dynamo_disable()
    def score_candidates(
        self,
        *,
        belief_hidden: torch.Tensor,  # [B, S, H]
        candidate_tokens: torch.Tensor,  # [B, S, A, E]
        candidate_valid: torch.Tensor,  # [B, S, A]
    ) -> torch.Tensor:
        if belief_hidden.ndim != 3:
            raise RuntimeError(f"belief_hidden must be [B, S, H], got {tuple(belief_hidden.shape)}")
        if candidate_tokens.ndim != 4:
            raise RuntimeError(
                f"candidate_tokens must be [B, S, A, E], got {tuple(candidate_tokens.shape)}"
            )
        if candidate_valid.ndim != 3:
            raise RuntimeError(f"candidate_valid must be [B, S, A], got {tuple(candidate_valid.shape)}")
        if candidate_tokens.shape[:3] != candidate_valid.shape:
            raise RuntimeError(
                "candidate_tokens/candidate_valid mismatch: "
                f"{tuple(candidate_tokens.shape[:3])} vs {tuple(candidate_valid.shape)}"
            )
        if candidate_tokens.shape[:2] != belief_hidden.shape[:2]:
            raise RuntimeError(
                "candidate_tokens/belief_hidden mismatch: "
                f"{tuple(candidate_tokens.shape[:2])} vs {tuple(belief_hidden.shape[:2])}"
            )
        if candidate_tokens.shape[3] != self.max_belief_tokens:
            raise RuntimeError(
                "candidate_tokens token width mismatch: "
                f"{candidate_tokens.shape[3]} vs {self.max_belief_tokens}"
            )
        if candidate_tokens.shape[2] != self.max_belief_candidates:
            raise RuntimeError(
                "candidate_tokens candidate-axis mismatch: "
                f"{candidate_tokens.shape[2]} vs {self.max_belief_candidates}"
            )

        batch_size, rows, num_candidates, row_width = candidate_tokens.shape
        flat_tokens = candidate_tokens.reshape(batch_size * rows, num_candidates, row_width)
        flat_mask = candidate_valid.to(dtype = torch.bool).reshape(batch_size * rows, num_candidates)
        encoded_flat = self._encode_sparse_candidates_eager(
            flat_tokens = flat_tokens,
            flat_mask = flat_mask,
        )
        candidate_hidden = encoded_flat.reshape(
            batch_size,
            rows,
            num_candidates,
            encoded_flat.shape[-1],
        )
        fused_candidate_state = belief_hidden.unsqueeze(-2) * candidate_hidden
        logits = self.candidate_logit_head(fused_candidate_state).squeeze(-1)
        invalid = ~candidate_valid.to(dtype = torch.bool)
        min_value = torch.finfo(logits.dtype).min
        return logits.masked_fill(invalid, min_value)

    def forward(
        self,
        *,
        obs_prefix_tokens: torch.Tensor,  # [B, T_obs, L_obs]
        obs_prefix_mask: torch.Tensor,  # [B, T_obs]
        state_channels: torch.Tensor,  # [B, C]
        belief_input_tokens: torch.Tensor,  # [B, S, E]
        candidate_tokens: torch.Tensor,  # [B, S, A, E]
        candidate_valid: torch.Tensor,  # [B, S, A]
        reward_to_go_input_tokens: torch.Tensor | None = None,  # [B, S + 1, E]
    ) -> BeliefTokenOutput:
        if obs_prefix_tokens.ndim != 3:
            raise RuntimeError(
                f"obs_prefix_tokens must be [B, T_obs, L_obs], got {tuple(obs_prefix_tokens.shape)}"
            )
        if obs_prefix_mask.shape != obs_prefix_tokens.shape[:2]:
            raise RuntimeError(
                f"obs_prefix_mask mismatch: {tuple(obs_prefix_mask.shape)} vs {tuple(obs_prefix_tokens.shape[:2])}"
            )
        if state_channels.ndim != 2 or state_channels.shape[0] != obs_prefix_tokens.shape[0]:
            raise RuntimeError(
                f"state_channels must be [B, C], got {tuple(state_channels.shape)} for B={obs_prefix_tokens.shape[0]}"
            )
        if belief_input_tokens.ndim != 3:
            raise RuntimeError(
                f"belief_input_tokens must be [B, S, E], got {tuple(belief_input_tokens.shape)}"
            )
        if belief_input_tokens.shape[2] != self.max_belief_tokens:
            raise RuntimeError(
                "belief_input_tokens token width mismatch: "
                f"{belief_input_tokens.shape[2]} vs {self.max_belief_tokens}"
            )
        if belief_input_tokens.shape[1] != self.max_belief_rows:
            raise RuntimeError(
                "belief_input_tokens row mismatch: "
                f"{belief_input_tokens.shape[1]} vs {self.max_belief_rows}"
            )
        if reward_to_go_input_tokens is not None:
            if reward_to_go_input_tokens.ndim != 3:
                raise RuntimeError(
                    "reward_to_go_input_tokens must be [B, S + 1, E], "
                    f"got {tuple(reward_to_go_input_tokens.shape)}"
                )
            expected_reward_input_shape = (
                int(obs_prefix_tokens.shape[0]),
                self.max_belief_rows + 1,
                self.max_belief_tokens,
            )
            if tuple(reward_to_go_input_tokens.shape) != expected_reward_input_shape:
                raise RuntimeError(
                    "reward_to_go_input_tokens shape mismatch: "
                    f"{tuple(reward_to_go_input_tokens.shape)} vs {expected_reward_input_shape}"
                )

        obs_rows = self._pool_rows(self.obs_embed, obs_prefix_tokens)
        if reward_to_go_input_tokens is None:
            decoder_belief_rows = self._encode_belief_input_rows(belief_input_tokens)
        else:
            decoder_belief_rows = self._encode_belief_input_rows(reward_to_go_input_tokens)
        state_row = self.state_mlp(state_channels.to(dtype = torch.float32)).unsqueeze(1)

        batch_size, obs_rows_len = obs_rows.shape[:2]
        if self.reward_to_go_head is not None and reward_to_go_input_tokens is not None:
            assert self.reward_to_go_query is not None
            reward_query_row = self.reward_to_go_query.to(dtype = decoder_belief_rows.dtype).view(1, 1, -1).expand(
                batch_size,
                1,
                -1,
            )
            decoder_belief_rows = torch.cat([decoder_belief_rows, reward_query_row], dim = 1)
        decoder_belief_rows_len = decoder_belief_rows.shape[1]
        full_rows = torch.cat([obs_rows, state_row, decoder_belief_rows], dim = 1)

        seg_obs = torch.zeros((batch_size, obs_rows_len), dtype = torch.long, device = full_rows.device)
        seg_state = torch.ones((batch_size, 1), dtype = torch.long, device = full_rows.device)
        seg_belief = torch.full(
            (batch_size, decoder_belief_rows_len),
            2,
            dtype = torch.long,
            device = full_rows.device,
        )
        segment_ids = torch.cat([seg_obs, seg_state, seg_belief], dim = 1)
        full_rows = full_rows + self.segment_embed(segment_ids)

        attn_mask = torch.cat(
            [
                obs_prefix_mask.to(dtype = torch.long),
                torch.ones((batch_size, 1), dtype = torch.long, device = full_rows.device),
                torch.ones((batch_size, decoder_belief_rows_len), dtype = torch.long, device = full_rows.device),
            ],
            dim = 1,
        )

        hidden = self.decoder(
            inputs_embeds = full_rows,
            attention_mask = attn_mask,
            use_cache = False,
            return_dict = True,
        ).last_hidden_state
        belief_hidden = hidden[:, obs_rows_len + 1 : obs_rows_len + 1 + self.max_belief_rows, :]
        logits = self.score_candidates(
            belief_hidden = belief_hidden,
            candidate_tokens = candidate_tokens,
            candidate_valid = candidate_valid,
        )
        reward_to_go = None
        if self.reward_to_go_head is not None and reward_to_go_input_tokens is not None:
            reward_hidden = hidden[:, -1, :]
            reward_to_go = self.reward_to_go_head(reward_hidden).squeeze(-1)
        return BeliefTokenOutput(
            logits = logits,
            belief_hidden = belief_hidden,
            full_hidden = hidden,
            reward_to_go = reward_to_go,
        )
