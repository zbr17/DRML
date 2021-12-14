from gedml.core.collectors import BaseCollector

class DefaultCollector(BaseCollector):
    """
    This is the default collector which directly computes metric matrix using embeddings.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, embeddings, labels) -> tuple:
        """
        Do nothing. Copy embeddings as proxies and copy labels as proxies labels.
        """
        proxies = embeddings
        proxies_labels = labels
        metric_mat = self.metric(embeddings, proxies)
        is_same_source = True
        return (
            metric_mat, 
            labels.unsqueeze(-1),
            proxies_labels.unsqueeze(0),
            is_same_source
        )