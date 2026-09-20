from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

class MessagePassing(nn.Module):

    def __init__(self, width: int):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(()))
        self.mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width), nn.LayerNorm(width), nn.SiLU())

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.mlp((1 + self.eps) * x + torch.bmm(adj, x) / adj.shape[-1])

class GraphCVAE(nn.Module):

    def __init__(self, n: int=20, hidden: int=32, latent: int=12, families: int=2, layers: int=3):
        super().__init__()
        self.n, self.hidden, self.latent, self.families, self.layers_count = (n, hidden, latent, families, layers)
        self.input = nn.Linear(2, hidden)
        self.message_layers = nn.ModuleList([MessagePassing(hidden) for _ in range(layers)])
        self.posterior = nn.Sequential(nn.Linear(2 * hidden + families, hidden), nn.SiLU(), nn.Linear(hidden, 2 * latent))
        self.decoder = nn.Sequential(nn.Linear(latent + families, 2 * hidden), nn.SiLU(), nn.Linear(2 * hidden, 2 * hidden), nn.SiLU(), nn.Linear(2 * hidden, n * (n - 1) // 2))
        self.register_buffer('edge_index', torch.triu_indices(n, n, offset=1))

    def config(self) -> dict:
        return {'n': self.n, 'hidden': self.hidden, 'latent': self.latent, 'families': self.families, 'layers': self.layers_count}

    def encode(self, adj: torch.Tensor, family: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        degree = adj.sum(-1, keepdim=True) / (self.n - 1)
        x = F.silu(self.input(torch.cat([torch.ones_like(degree), degree], dim=-1)))
        for layer in self.message_layers:
            x = x + layer(x, adj)
        pooled = torch.cat([x.mean(1), x.amax(1), F.one_hot(family, self.families).float()], dim=-1)
        mean, logvar = self.posterior(pooled).chunk(2, dim=-1)
        return (mean, logvar.clamp(-8, 6))

    def decode(self, z: torch.Tensor, family: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([z, F.one_hot(family, self.families).float()], dim=-1))

    def forward(self, adj: torch.Tensor, family: torch.Tensor, sample: bool=True):
        mean, logvar = self.encode(adj, family)
        z = mean + torch.randn_like(mean) * (0.5 * logvar).exp() if sample else mean
        return (self.decode(z, family), mean, logvar)

    def loss(self, adj: torch.Tensor, family: torch.Tensor, beta: float=0.01, structural_weight: float=0.2, sample: bool=True):
        logits, mean, logvar = self(adj, family, sample)
        target = adj[:, self.edge_index[0], self.edge_index[1]]
        reconstruction = F.binary_cross_entropy_with_logits(logits, target)
        kl = 0.5 * (mean.square() + logvar.exp() - 1 - logvar).mean()
        probabilities = torch.zeros_like(adj)
        probabilities[:, self.edge_index[0], self.edge_index[1]] = logits.sigmoid()
        probabilities = probabilities + probabilities.transpose(1, 2)
        degree = F.mse_loss(probabilities.sum(-1) / (self.n - 1), adj.sum(-1) / (self.n - 1))
        total = reconstruction + beta * kl + structural_weight * degree
        return (total, {'loss': float(total.detach()), 'bce': float(reconstruction.detach()), 'kl_per_dim': float(kl.detach()), 'degree_loss': float(degree.detach())})

def parameter_counts(model: GraphCVAE) -> dict[str, int]:
    total = sum((p.numel() for p in model.parameters()))
    decoder = sum((p.numel() for p in model.decoder.parameters()))
    return {'parameters_total': total, 'parameters_generation': decoder, 'parameter_bytes_fp32': total * 4, 'generation_parameter_bytes_fp32': decoder * 4}
import math
from numbers import Integral

class _ResidualMessageLayer(nn.Module):

    def __init__(self, hidden: int):
        super().__init__()
        self.self_map = nn.Linear(hidden, hidden)
        self.neighbor_map = nn.Linear(hidden, hidden, bias=False)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, nodes: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        aggregated = torch.bmm(adjacency, nodes) / adjacency.shape[-1]
        update = self.self_map(nodes) + self.neighbor_map(aggregated)
        return nodes + F.silu(self.norm(update))

class EdgeDenoiser(nn.Module):

    def __init__(self, n: int=20, hidden: int=32, families: int=2, layers: int=3, use_common_neighbors: bool=True):
        super().__init__()
        for name, value, minimum in (('n', n, 2), ('hidden', hidden, 1), ('families', families, 1), ('layers', layers, 0)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        self.n = int(n)
        self.hidden = int(hidden)
        self.families = int(families)
        self.layers_count = int(layers)
        self.use_common_neighbors = bool(use_common_neighbors)
        self.condition = nn.Sequential(nn.Linear(families + 4, hidden), nn.SiLU())
        self.node_input = nn.Linear(2, hidden)
        self.message_layers = nn.ModuleList([_ResidualMessageLayer(hidden) for _ in range(layers)])
        scalar_features = 4 if self.use_common_neighbors else 3
        edge_feature_count = 4 * hidden + scalar_features
        self.edge_mlp = nn.Sequential(nn.Linear(edge_feature_count, 2 * hidden), nn.SiLU(), nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.register_buffer('edge_index', torch.triu_indices(n, n, offset=1))

    def config(self) -> dict:
        return {'n': self.n, 'hidden': self.hidden, 'families': self.families, 'layers': self.layers_count, 'use_common_neighbors': self.use_common_neighbors}

    def forward(self, noisy_adj: torch.Tensor, family: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        if noisy_adj.ndim != 3 or noisy_adj.shape[-2:] != (self.n, self.n):
            raise ValueError(f'noisy_adj must have shape [B,{self.n},{self.n}]')
        batch_size = noisy_adj.shape[0]
        if family.shape != (batch_size,) or time.shape != (batch_size,):
            raise ValueError('family and time must have shape [B]')
        time = time.to(dtype=noisy_adj.dtype)
        time_features = torch.stack((time, time.square(), torch.sin(math.pi * time), torch.cos(math.pi * time)), dim=-1)
        condition = self.condition(torch.cat((F.one_hot(family, self.families).to(noisy_adj.dtype), time_features), dim=-1))
        degree = noisy_adj.sum(dim=-1) / (self.n - 1)
        nodes = F.silu(self.node_input(torch.stack((torch.ones_like(degree), degree), dim=-1)) + condition[:, None, :])
        for layer in self.message_layers:
            nodes = layer(nodes, noisy_adj)
        left, right = self.edge_index
        node_left, node_right = (nodes[:, left], nodes[:, right])
        scalar_parts = [noisy_adj[:, left, right], degree[:, left] + degree[:, right], (degree[:, left] - degree[:, right]).abs()]
        if self.use_common_neighbors:
            common = torch.bmm(noisy_adj, noisy_adj) / self.n
            scalar_parts.append(common[:, left, right])
        edge_features = torch.cat((node_left + node_right, (node_left - node_right).abs(), node_left * node_right, torch.stack(scalar_parts, dim=-1), condition[:, None, :].expand(-1, left.shape[0], -1)), dim=-1)
        return self.edge_mlp(edge_features).squeeze(-1)

def alpha_bar(time: torch.Tensor, total_steps: int=32) -> torch.Tensor:
    if isinstance(total_steps, bool) or not isinstance(total_steps, Integral) or total_steps < 1:
        raise ValueError('total_steps must be a positive integer')
    value = torch.as_tensor(time)
    if not value.is_floating_point():
        value = value.to(torch.float32)
    if not torch.isfinite(value).all() or (value < 0).any() or (value > total_steps).any():
        raise ValueError('time must be finite and between zero and total_steps')
    schedule = torch.cos(value * (math.pi / (2 * total_steps))).square()
    schedule = torch.where(value == 0, torch.ones_like(schedule), schedule)
    schedule = torch.where(value == total_steps, torch.zeros_like(schedule), schedule)
    return schedule

def _validate_prior(edge_prior: float) -> float:
    prior = float(edge_prior)
    if not math.isfinite(prior) or not 0 < prior < 1:
        raise ValueError('edge_prior must be strictly between zero and one')
    return prior

def reverse_probability(predicted_clean_probability: torch.Tensor, noisy_edges: torch.Tensor, alpha_s: float, alpha_t: float, edge_prior: float) -> torch.Tensor:
    prior = _validate_prior(edge_prior)
    retained_s, retained_t = (float(alpha_s), float(alpha_t))
    if not all((math.isfinite(value) for value in (retained_s, retained_t))):
        raise ValueError('cumulative coefficients must be finite')
    if not 0 < retained_s <= 1 or not 0 <= retained_t <= retained_s:
        raise ValueError('require 0 < alpha_s <= 1 and 0 <= alpha_t <= alpha_s')
    prediction = predicted_clean_probability
    if prediction.ndim != 2 or prediction.shape != noisy_edges.shape:
        raise ValueError('prediction and noisy_edges must have the same [B,E] shape')
    if not prediction.is_floating_point():
        raise ValueError('predicted probabilities must be floating point')
    if prediction.device != noisy_edges.device:
        raise ValueError('prediction and noisy_edges must be on the same device')
    if not torch.isfinite(prediction).all() or (prediction < 0).any() or (prediction > 1).any():
        raise ValueError('predicted probabilities must be finite and in [0,1]')
    if not ((noisy_edges == 0) | (noisy_edges == 1)).all():
        raise ValueError('noisy_edges must contain binary states')
    observed = noisy_edges.to(dtype=prediction.dtype)
    if retained_s == retained_t:
        return observed.clone()
    if retained_s == 1.0:
        return prediction.clone()
    interval_retention = retained_t / retained_s
    observed_prior = observed * prior + (1 - observed) * (1 - prior)
    likelihood_if_one = interval_retention * observed + (1 - interval_retention) * observed_prior
    likelihood_if_zero = interval_retention * (1 - observed) + (1 - interval_retention) * observed_prior
    previous_one_if_clean_zero = (1 - retained_s) * prior
    previous_one_if_clean_one = retained_s + previous_one_if_clean_zero

    def conditional_posterior(previous_one: float) -> torch.Tensor:
        numerator = likelihood_if_one * previous_one
        denominator = numerator + likelihood_if_zero * (1 - previous_one)
        return numerator / denominator.clamp_min(torch.finfo(prediction.dtype).tiny)
    posterior_if_zero = conditional_posterior(previous_one_if_clean_zero)
    posterior_if_one = conditional_posterior(previous_one_if_clean_one)
    result = (1 - prediction) * posterior_if_zero + prediction * posterior_if_one
    return result.clamp(0, 1)

def corrupt_edges(clean_edges: torch.Tensor, time_int_tensor: torch.Tensor, steps: int, edge_prior: float, generator: torch.Generator | None=None) -> torch.Tensor:
    prior = _validate_prior(edge_prior)
    if clean_edges.ndim != 2 or not clean_edges.is_floating_point():
        raise ValueError('clean_edges must be a floating [B,E] tensor')
    if not ((clean_edges == 0) | (clean_edges == 1)).all():
        raise ValueError('clean_edges must contain binary states')
    times = torch.as_tensor(time_int_tensor, device=clean_edges.device)
    if times.ndim != 1 or len(times) != len(clean_edges):
        raise ValueError('time_int_tensor must have one integer time per graph')
    if times.is_floating_point() and (not (times == times.round()).all()):
        raise ValueError('corruption times must be integers')
    retained = alpha_bar(times.to(dtype=clean_edges.dtype), total_steps=steps)[:, None]
    probabilities = retained * clean_edges + (1 - retained) * prior
    return torch.bernoulli(probabilities, generator=generator)
