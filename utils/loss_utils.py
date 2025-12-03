import torch
import torch.nn.functional as F


def get_activation_dit(activation_dict, name):
    """
    Helper function to create a forward hook.

    This function returns a hook that stores the output of a module
    in the provided dictionary.
    
    Args:
        activation_dict (dict): The dictionary to store the activations.
        name (str): The key under which to store the activation.
        
    Returns:
        function: A forward hook function.
    """
    def hook(model, input, output):
        # The output of a DiTBlock is a single tensor, so we store it directly.
        activation_dict[name] = output
    return hook


def add_block_hooks_dit(model, activation_dict):
    """
    Adds a forward hook only to the middle DiTBlock in the DiT model.

    Args:
        model (nn.parallel.DistributedDataParallel): The DDP-wrapped DiT model.
        activation_dict (dict): Dictionary to store activations.
    """
    unwrapped_model = model.module
    num_blocks = len(unwrapped_model.blocks)
    
    middle_idx = num_blocks // 2
    hook_name = f'blocks.{middle_idx}'
    unwrapped_model.blocks[middle_idx].register_forward_hook(get_activation_dit(activation_dict, hook_name))


def add_block_hooks_sd3(accelerator, model, activation_dict):
    """
    Adds a forward hook only to the middle transformer block in the SD3 model.

    Args:
        accelerator (Accelerator): The accelerator object.
        model (nn.parallel.DistributedDataParallel): The DDP-wrapped SD3 model.
        activation_dict (dict): Dictionary to store activations.
    """
    model = accelerator.unwrap_model(model)
    num_blocks = len(model.transformer_blocks)
    middle_idx = num_blocks // 2
    hook_name = f'transformer_blocks.{middle_idx}'
    model.transformer_blocks[middle_idx].register_forward_hook(get_activation_dit(activation_dict, hook_name))  


def get_activation(activation, name, residual=False):
    if residual:
        def get_output_hook(module, input, output):
            activation[name] = output[0]
    else:
        def get_output_hook(module, input, output):
            activation[name] = output
    return get_output_hook


def add_block_hooks(accelerator, unet, dicts):
    unet = accelerator.unwrap_model(unet)
    for i in range(len(unet.down_blocks)):
        unet.down_blocks[i].register_forward_hook(get_activation(dicts, 'down_blocks.'+str(i), True))
    unet.mid_block.register_forward_hook(get_activation(dicts, 'mid_block', False))
    for i in range(len(unet.up_blocks)):
        unet.up_blocks[i].register_forward_hook(get_activation(dicts, 'up_blocks.'+str(i), False))


def router_balance_loss(logits, samples, n_experts):
    """
    logits:  [T, n_experts]  (raw router outputs before Gumbel-softmax)
    samples: [T, n_experts]  (the one-hot or soft assignments from Gumbel-softmax)
    n_experts: int
    """
    # 1) Compute P_i: average softmax probability allocated to expert i
    p = F.softmax(logits, dim=-1).mean(dim=0)  # shape [n_experts]

    # 2) Compute F_i: fraction of tokens actually assigned to expert i
    chosen_experts = samples.argmax(dim=1)
    sample_hard = sample_hard = torch.nn.functional.one_hot(chosen_experts, num_classes=n_experts).float().to(logits.device)
    f = sample_hard.mean(dim=0)  # shape [n_experts]

    # 3) Load-balancing regularization: R_L = N * sum_i (F_i * P_i)
    r_L = n_experts * torch.sum(p * f)
    return r_L


def orthogonality_loss(expert_feat):
    """
    expert_feat: Tensor shape [n_experts, n_layers, feat_dim]
    returns: scalar loss representing average cosine similarity among flattened expert vectors
    """
    if len(expert_feat.shape) == 3:
        n_experts, n_layers, feat_dim = expert_feat.shape
        # Flatten each expert's layers into a single vector
        expert_feat = expert_feat.reshape(n_experts, -1)  # [n_experts, n_layers * feat_dim]
    else:
        n_experts, feat_dim = expert_feat.shape

    # Normalize flattened features
    expert_flat_norm = expert_feat / (expert_feat.norm(dim=-1, keepdim=True) + 1e-8)  # [n_experts, flattened_dim]
    cos_sim_matrix = torch.matmul(expert_flat_norm, expert_flat_norm.T)

    # Exclude diagonal (self-similarity)
    identity = torch.eye(n_experts, device=expert_feat.device)
    cos_sim_off_diag = cos_sim_matrix = cos_sim_matrix * (1 - identity)

    # Compute average similarity (excluding diagonal pairs)
    loss = cos_sim_off_diag.sum() / (n_experts * (n_experts - 1))

    return loss


def kl_diversity_loss(expert_feat):
    """
    Compute a symmetric KL-divergence diversity loss among experts using vectorized operations.
    expert_feat: Tensor of shape [n_experts, n_layers] representing logits.
    Returns a scalar diversity loss value.
    """
    n_experts, n_layers = expert_feat.shape
    if n_experts <= 1:
        return torch.tensor(0.0, device=expert_feat.device, dtype=expert_feat.dtype)

    log_probs = F.log_softmax(expert_feat, dim=-1)  # [n_experts, n_layers]
    probs = log_probs.exp()                         # [n_experts, n_layers]

    # Expand dimensions to compute pairwise KL divergences:
    # - probs_i and log_probs_i have shape [n_experts, 1, n_layers]
    # - log_probs_j has shape [1, n_experts, n_layers]
    probs_i = probs.unsqueeze(1)
    log_probs_i = log_probs.unsqueeze(1)
    log_probs_j = log_probs.unsqueeze(0)

    # Compute pairwise KL divergence: KL(p_i || p_j)
    kl_matrix = torch.sum(probs_i * (log_probs_i - log_probs_j), dim=-1) / n_layers  # Shape: [n_experts, n_experts]

    # Compute the symmetric KL divergence between each pair: 0.5*(KL(p_i||p_j) + KL(p_j||p_i))
    sym_kl_matrix = 0.5 * (kl_matrix + kl_matrix.transpose(0, 1))

    # Extract the upper-triangular (i < j) values to avoid redundant comparisons and self-comparison.
    upper_tri_indices = torch.triu_indices(n_experts, n_experts, offset=1)
    pairwise_sym_kl = sym_kl_matrix[upper_tri_indices[0], upper_tri_indices[1]]
    diversity_loss = pairwise_sym_kl.mean()
    return diversity_loss


# the value matching loss
# **usage:    match_loss(x, y = target_value)
# **function: make x --> closer to target_value
def match_loss(x, y, epsilon=1e-8):
    """
    Computes R(x, y) = log(max(x, y) / (min(x, y) + epsilon)) to avoid division by zero.
    
    Args:
    x (torch.Tensor): Input tensor x.
    y (torch.Tensor): Input tensor y.
    epsilon (float): Small constant for numerical stability to prevent division by zero.
    
    Returns:
    torch.Tensor: The computed loss value.
    """
    # Compute max(x, y) and min(x, y)
    if torch.is_tensor(y):
        y = y.to(dtype=x.dtype, device=x.device)
    else:
        y = torch.tensor(y, dtype=x.dtype, device=x.device)
    max_val = torch.max(x, y)
    min_val = torch.min(x, y)

    # To avoid division by zero, add a small epsilon to min_val
    ratio = max_val / (min_val + epsilon)

    # Compute log(max / min)
    loss = torch.log(ratio)
    
    return loss

