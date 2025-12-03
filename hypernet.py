import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    #U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_sigmoid(logits, T, offset=0, hard=False):
    gumbel_sample = sample_gumbel(logits.size())
    if logits.is_cuda:
        gumbel_sample = gumbel_sample.cuda()

    y = logits + gumbel_sample + offset
    y = torch.sigmoid(y / T)

    if hard:
        y_hard = torch.zeros(y.size(), device=logits.device)
        y_hard[y>=0.5]=1
        y_hard = (y_hard - y).detach() + y
        return y_hard
    return y


def gumbel_softmax(logits, T, offset=0, hard=False):
    gumbel_sample = sample_gumbel(logits.size())
    if logits.is_cuda:
        gumbel_sample = gumbel_sample.cuda()
    
    y = logits + gumbel_sample + offset
    y = F.softmax(y / T, dim=-1)

    if hard:
        y_hard = torch.zeros(y.size(), device=logits.device)
        # Get argmax indices along the last dimension
        argmax_indices = torch.argmax(y, dim=-1, keepdim=True)
        y_hard.scatter_(dim=-1, index=argmax_indices, value=1)
        y_hard = (y_hard - y).detach() + y
        return y_hard
    return y


class Expert_Router(nn.Module):
    def __init__(self, time_embedding, n_experts, hidden_dim=64, T=0.4, base=0):
        super(Expert_Router, self).__init__()
        self.n_experts = n_experts
        self.T = T         # Gumbel Softmax temperature
        self.base = base   # Offset for sampling

        # Register the fixed time_embedding
        # time_embedding should have shape [n_timesteps, time_emb_dim]
        self.register_buffer('time_embedding', torch.tensor(time_embedding, dtype=torch.float32))

        # Create layers using the time_embedding's dimension as input
        self.linear1 = nn.Linear(self.time_embedding.size(1), hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, n_experts, bias=False)

    def forward(self, x=None):
        # Use the fixed time_embedding as input.
        outputs = self.linear1(self.time_embedding)  # [n_timesteps, hidden_dim]
        outputs = F.relu(self.ln(outputs))
        outputs = self.linear2(outputs)  # [n_timesteps, n_experts]

        # Apply Gumbel Softmax sampling
        out = gumbel_softmax(outputs, T=self.T, offset=self.base, hard=(not self.training))  # [timesteps, n_experts]

        return outputs, out  # Shape: [timesteps, n_experts]
    

class SD_HyperStructure(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, time_embedding, n_intervals, p_structures, T=0.4, base=3, input_dim=64, hidden_dim=256):
        super(SD_HyperStructure, self).__init__()
        self.n_layers = sum(itertools.chain.from_iterable(p_structures))
        self.p_structures = p_structures
        self.T = T
        self.base = base
        self.n_intervals = n_intervals
        self.out = None

        self.register_buffer("inputs", torch.Tensor(self.n_intervals, self.n_layers, input_dim))
        nn.init.orthogonal_(self.inputs)

        self.router = Expert_Router(time_embedding, n_intervals, hidden_dim=input_dim)
        # Define the hypernet
        self.ln = nn.LayerNorm(input_dim)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x=None):
        outputs = self.linear1(self.ln(self.inputs))  # [n_experts, n_layers, 256]
        outputs = F.relu(self.ln2(outputs))
        outputs = self.linear2(outputs).squeeze(-1)  # outputs: [n_experts, n_layers]
        # Apply Gumbel sigmoid sampling
        expert_logits = gumbel_sigmoid(outputs, T=self.T, offset=self.base, hard=(not self.training))  # [n_experts, n_layers]

        # Get the router's selection probabilities
        router_logits, router_out = self.router()  # [n_timesteps, n_experts]
        out = torch.einsum("te,el->tl", router_out, expert_logits)  # [n_timesteps, n_layers]
            
        self.out = out.detach()
        return out, router_out, router_logits, expert_logits  # Shape: [T, n_layers]
    
    def get_timestep_expert(self, timestep):
        cur_expert = self.out[timestep]
        return self.transform_output(cur_expert)
    
    def transform_output(self, input):
        """
        input_tensor: input: shape (T, n_layers) or possibly (n_layers,) if T=1
        Returns a list of "arch_vectors",
        each arch_vector is shaped according to self.p_structures.
        """
        if input.dim() == 1: 
            # If it's shape (n_layers,), make it (1, n_layers)
            input = input.unsqueeze(0)

        # arch_vector share the same structure with p_structures
        split_sizes = [sum(p) for p in self.p_structures]
        splitted_rows = torch.split(input, split_sizes, dim=1)

        arch_vectors = []
        for i, structure in enumerate(self.p_structures):
            chunk_i = splitted_rows[i] # (T, sum(structure))
            subpieces = torch.split(chunk_i, structure, dim=1)
            arch_vectors.append(list(subpieces))

        return arch_vectors
    

class DiT_HyperStructure(nn.Module):
    def __init__(self, time_embedding, n_intervals, n_layers, T=0.4, base=3, input_dim=64, hidden_dim=256):
        super(DiT_HyperStructure, self).__init__()
        self.n_layers = n_layers
        self.n_intervals = n_intervals
        self.T = T
        self.base = base

        self.register_buffer("inputs", torch.Tensor(n_intervals, n_layers, input_dim))
        nn.init.orthogonal_(self.inputs)

        self.router = Expert_Router(time_embedding, n_intervals, hidden_dim=input_dim)
        # Define the hypernet
        self.ln = nn.LayerNorm(input_dim)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x=None):
        outputs = self.linear1(self.ln(self.inputs))  # [n_experts, n_layers, 256]
        outputs = F.relu(self.ln2(outputs))
        outputs = self.linear2(outputs).squeeze(-1)  # outputs: [n_experts, n_layers]
        # Apply Gumbel sigmoid sampling
        expert_logits = gumbel_sigmoid(outputs, T=self.T, offset=self.base, hard=(not self.training))
        # Get the router's selection probabilities
        router_logits, router_out = self.router()  # [n_timesteps, n_experts]
        out = torch.einsum("te,el->tl", router_out, expert_logits)  # [n_timesteps, n_layers]
        
        return out, router_out, router_logits, expert_logits
