"""
Entity-transformer actor-critic for Stable-Baselines3.

The flat observation (see ``atc_agent.features``) is split back into tokens:
one ego token, up to M_AC aircraft tokens and N_OBS obstacle tokens, each
embedded by its own MLP plus a learned type embedding. A pre-LayerNorm
Transformer encoder with key-padding masks lets the ego token attend to the
variable set of traffic / obstacles (permutation invariant, handles any number
of present entities, shared between the single- and multi-agent tasks).
Features = [ego token output, masked mean over all tokens].
"""

import numpy as np
import torch as th
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from atc_agent.actions import HEADING_OFFSETS, SPEED_TARGETS
from atc_agent.features import (
    EGO_DIM, AC_DIM, OB_DIM, M_AC, N_OBS, AC_OFF, AC_MASK_OFF, OB_OFF, OB_MASK_OFF,
)


class EntityTransformer(BaseFeaturesExtractor):
    def __init__(self, observation_space, d_model=128, n_heads=4, n_layers=2, ff_mult=2, amp=True):
        super().__init__(observation_space, features_dim=2 * d_model)
        self.amp = amp      # bf16 autocast on CUDA (~2x faster updates); fp32 on CPU

        def embed(n_in):
            return nn.Sequential(nn.Linear(n_in, d_model), nn.GELU(), nn.Linear(d_model, d_model))

        self.ego = embed(EGO_DIM)
        self.ac = embed(AC_DIM)
        self.ob = embed(OB_DIM)
        self.type_emb = nn.Parameter(th.zeros(3, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, ff_mult * d_model, dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        use_amp = self.amp and x.is_cuda and th.cuda.is_bf16_supported()
        with th.autocast(device_type="cuda", dtype=th.bfloat16, enabled=use_amp):
            out = self._encode(x)
        return out.float()

    def _encode(self, x):
        B = x.shape[0]
        ego = x[:, :EGO_DIM]
        ac = x[:, AC_OFF:AC_MASK_OFF].reshape(B, M_AC, AC_DIM)
        ac_ok = x[:, AC_MASK_OFF:OB_OFF] > 0.5
        ob = x[:, OB_OFF:OB_MASK_OFF].reshape(B, N_OBS, OB_DIM)
        ob_ok = x[:, OB_MASK_OFF:] > 0.5
        tok = th.cat([
            self.ego(ego)[:, None] + self.type_emb[0],
            self.ac(ac) + self.type_emb[1],
            self.ob(ob) + self.type_emb[2],
        ], dim=1)
        valid = th.cat([th.ones(B, 1, dtype=th.bool, device=x.device), ac_ok, ob_ok], dim=1)
        h = self.norm(self.encoder(tok, src_key_padding_mask=~valid))
        w = valid.unsqueeze(-1).to(h.dtype)
        pooled = (h * w).sum(1) / w.sum(1)
        return th.cat([h[:, 0], pooled], dim=-1)


class AtcPolicy(ActorCriticPolicy):
    """ActorCriticPolicy with the entity transformer and an initial action prior
    (logit bias favouring direct-to-goal and fast cruise; still fully learned)."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("features_extractor_class", EntityTransformer)
        kwargs.setdefault("share_features_extractor", False)
        kwargs.setdefault("net_arch", dict(pi=[256, 256], vf=[256, 256]))
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        nh = len(HEADING_OFFSETS)
        prior = np.concatenate([
            -np.abs(HEADING_OFFSETS) / 30.0,
            np.linspace(-1.5, 0.6, len(SPEED_TARGETS)),
        ])
        with th.no_grad():
            self.action_net.bias.copy_(th.as_tensor(prior, dtype=self.action_net.bias.dtype))
        assert self.action_net.bias.shape[0] == nh + len(SPEED_TARGETS)
