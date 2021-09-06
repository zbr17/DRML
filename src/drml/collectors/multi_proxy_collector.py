import torch
import torch.nn as nn 
import math

from gedml.core.collectors import BaseCollector

class MultiProxyCollector(BaseCollector):
    def __init__(
        self,
        num_classes=100,
        embedding_size=512,
        proxy_group=4,
        *args,
        **kwargs,
    ):
        super(MultiProxyCollector, self).__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.proxy_group = proxy_group

        # initaite params
        self.initiate_params()
    
    def initiate_params(self):
        """
        Initiate proxies.
        """
        total_num = self.num_classes
        # define proxies
        for i in range(self.proxy_group):
            setattr(
                self,
                "proxy_{}".format(i),
                torch.nn.Parameter(
                    torch.randn(total_num, self.embedding_size)
                )
            )

        # initiate proxies
        for i in range(self.proxy_group):
            proxy = getattr(self, "proxy_{}".format(i))
            nn.init.kaiming_normal_(proxy.data, a=math.sqrt(5))

        # initiate labels
        proxy_labels = torch.arange(self.num_classes)
        self.register_buffer("labels", proxy_labels)
    
    def forward(self, data, embeddings: dict, labels: dict) -> torch.Tensor:
        assert isinstance(embeddings, (dict, list))
        assert isinstance(labels, (dict, list))

        output_list = []
        for i in range(len(embeddings)):
            cur_emb = embeddings[i]
            cur_label = labels[i]
            proxy = getattr(self, "proxy_{}".format(i))
            output_list.extend([
                self.compute_mat(cur_emb, proxy),
                cur_label.unsqueeze(1),
                self.labels.unsqueeze(0),
                False
            ])
        return tuple(output_list)
    
    def compute_mat(self, cur_emb, ref_emb=None):
        if ref_emb is None:
            ref_emb = self.proxies
        # normalize
        cur_emb = torch.nn.functional.normalize(cur_emb, dim=1, p=2)
        ref_emb = torch.nn.functional.normalize(ref_emb, dim=1, p=2)

        # compute metric matrix
        metric_mat = torch.matmul(cur_emb, ref_emb.t())
        return metric_mat

