import torch
from gedml.core.losses import BaseLoss
import torch.nn.functional as F 

class ZeroLoss(BaseLoss):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
    
    def required_metric(self):
        return ["cosine"]
    
    def compute_loss(
        self,
        metric_mat,
        row_labels,
        col_labels,
        is_same_source=False,
        *args,
        **kwargs
    ):
        loss = 0 * torch.sum(metric_mat)
        return loss
        