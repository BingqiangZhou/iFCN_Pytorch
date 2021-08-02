import torch

def modify_first_layer_weight(weights, first_layer_name='conv1', initialization_type='mean'):
    assert initialization_type in ['mean', 'zero']
    first_layer_weight_name = f'{first_layer_name}.weight'
    old_weight = weights[first_layer_weight_name] # [out_channel, in_channel, kernel_height, kernel_width]

    if initialization_type == 'mean':
        new_channel_weight = torch.mean(old_weight, dim=1, keepdim=True)
    else:
        out_channel, in_channel, kernel_height, kernel_width = old_weight.shape
        new_channel_weight = torch.zeros((out_channel, 1, kernel_height, kernel_width))
    
    new_weight = torch.cat([old_weight, new_channel_weight, new_channel_weight])
    weights[first_layer_weight_name] = new_weight
    
    return weights



