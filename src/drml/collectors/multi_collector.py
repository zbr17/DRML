import torch
import torch.nn as nn 
import math

from thudml.core.collectors import BaseCollector

class MultiCollector(BaseCollector):
    def __init__(
        self,
        proxy_group=4,
        *args,
        **kwargs,
    ):
        super(MultiCollector, self).__init__(*args, **kwargs)
        self.proxy_group = proxy_group
    
    def forward(self, data, embeddings: dict, labels: dict) -> torch.Tensor:
        assert isinstance(embeddings, (dict, list))
        assert isinstance(labels, (dict, list))

        output_list = []
        for i in range(len(embeddings)):
            cur_emb = embeddings[i]
            cur_label = labels[i]
            output_list.extend(
                [self.metric(cur_emb, cur_emb),
                cur_label.unsqueeze(1),
                cur_label.unsqueeze(0),
                True]
            )

        return tuple(output_list)
