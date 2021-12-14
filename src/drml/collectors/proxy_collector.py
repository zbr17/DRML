import torch
import math
from gedml.core.collectors import BaseCollector

class ProxyCollector(BaseCollector):
    def __init__(
        self,
        num_classes=100,
        embeddings_dim=128,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.embeddings_dim = embeddings_dim

        self.initiate_params()
    
    def initiate_params(self):
        """
        Initiate proxies.
        """
        self.proxies = torch.nn.Parameter(
            torch.randn(
                self.num_classes,
                self.embeddings_dim
            )
        )
        
        proxy_labels = torch.arange(self.num_classes)
        self.register_buffer("proxy_labels", proxy_labels)
        torch.nn.init.kaiming_uniform_(self.proxies, a=math.sqrt(5))
    
    def forward(self, embeddings, labels) -> tuple:
        """
        Compute similarity (or distance) matrix between embeddings and proxies.
        """
        metric_mat = self.metric(embeddings, self.proxies)
        is_same_source = False

        return (
            metric_mat,
            labels.unsqueeze(-1),
            self.proxy_labels.unsqueeze(0),
            is_same_source,
        )
