import torch
import torch.nn.functional as F

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

def upsample_size_to_target(to_upsample_tensor, target_tensor):
    _, _, h1, w1 =  to_upsample_tensor.shape
    _, _, h2, w2 =  target_tensor.shape
    if h1 != h2 or w1 != w2:
        out = F.interpolate(to_upsample_tensor, (h2, w2))
    return out

