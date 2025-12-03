
import re
from .op_counter import get_flops_pytorch


def select_pruning_mask(pruning_mask, interval_idx):
    """
    Args: pruning_mask: a nested list of lists of tensors. pruning_mask[i][j].shape = (T, piece_size)
        interval_idx: (bsz,) integer tensor with indices in [0..T-1].
    Returns: selected_arch: same nested-list structure, but each tensor is (bsz, piece_size).
    """
    selected_mask = []
    for i, subpieces in enumerate(pruning_mask):
        cur_mask = []
        for j, piece in enumerate(subpieces):
            selected_piece = piece[interval_idx]
            cur_mask.append(selected_piece)
        selected_mask.append(cur_mask)
    return selected_mask


def select_pruning_mask_dit(pruning_mask, interval_idx):
    """
    Args: pruning_mask: a list of tensors. pruning_mask[i].shape = (T, n_layers)
        interval_idx: (bsz,) integer tensor with indices in [0..T-1].
    Returns: selected_arch: same list structure, but each tensor is (bsz, n_layers).
    """
    return pruning_mask[interval_idx]


# the outputs share the same shape with p_structures
def pruning_ratio_contribution(model, input_sample):
    total_flops, total_params, results = get_flops_pytorch(model, input_sample)

    # get the corresponding blocks of the pruning structures
    transformer_blocks = r'^.*\.transformer_blocks\.\d+$'
    resnets = r'^.*\.resnets\.\d+$'
    keys = []
    for key in results.keys():
        if bool(re.match(transformer_blocks, key)) or bool(re.match(resnets, key)):
            keys.append(key)

    group_keys = group_keys_by_block(keys)

    # get the params and flops for these blocks
    flop_ratio = []
    param_ratio = []
    for keys in group_keys.values():
        cur_flop_ratio = []
        cur_param_ratio = []
        for k in keys:
            cur_result = results[k]
            cur_flop_ratio.append(cur_result['flops']/total_flops)
            cur_param_ratio.append(cur_result['params']/total_params)
        flop_ratio.append(cur_flop_ratio)
        param_ratio.append(cur_param_ratio)

    return flop_ratio, param_ratio


def group_keys_by_block(keys):
    group = {}
    for key in keys:
        # Split the key into parts
        parts = key.split('.')
        if parts[0] in ['down_blocks', 'up_blocks']:
            id = '.'.join(parts[:2])  # e.g., 'down_blocks.0'
        else:
            id = parts[0]  # e.g., 'mid_block'
        
        if id not in group:
            group[id] = []
        group[id].append(key)

    filtered_group = {}
    # Regular expressions to match transformer_blocks and resnets
    transformer_pattern = re.compile(r'attentions\.(\d+)\.transformer_blocks\.(\d+)')
    resnets_pattern = re.compile(r'resnets\.(\d+)$')
    # In each group, keep only the last resnet and the last transformer for a single attention
    for id, keys in group.items():
        attention_first_blocks = {}
        last_resnet = None
        last_resnet_index = -1

        for k in keys:
            # check for the transformer
            transformer_match = transformer_pattern.search(k)
            if transformer_match:
                attn_idx = int(transformer_match.group(1))
                block_idx = int(transformer_match.group(2))
                # If this is the first transformer_block for this attention index, keep it
                if attn_idx not in attention_first_blocks:
                    attention_first_blocks[attn_idx] = (k, block_idx)
                continue

            # Check for resnets
            resnets_match = resnets_pattern.search(k)
            if resnets_match:
                resnet_idx = int(resnets_match.group(1))
                # Update if this resnet_idx is greater than the current one
                if resnet_idx > last_resnet_index:
                    last_resnet = k
                    last_resnet_index = resnet_idx
                continue

        # Collect the filtered items
        filtered_items = [v[0] for v in attention_first_blocks.values()]
        if last_resnet:
            filtered_items.append(last_resnet)

        filtered_group[id] = filtered_items
        
    return filtered_group
