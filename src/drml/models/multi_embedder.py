import torch
import torch.nn as nn 
import torch.nn.functional as F

from gedml.core.modules import WithRecorder

class MultiEmbedder(WithRecorder):
    def __init__(
        self,
        embedding_size=128,
        embedder_group=1,
        input_size=256,
        div_gamma=1,
        weight_repre_loss=1,
    ):
        super(MultiEmbedder, self).__init__()

        self.embedding_size = embedding_size
        self.embedder_group = embedder_group
        self.input_size = input_size
        self.div_gamma = div_gamma
        self.weight_repre_loss = weight_repre_loss

        self.get_params()
        self.initiate_params()

        # to_record_list = [
        #     "num_branch_{}".format(i)
        #     for i in range(self.embedder_group)
        # ]
        # for item in to_record_list:
        #     self.add_recordable_attr(name=item)
        
    @property
    def output_list(self):
        return [
            "nodes_list",
            "labels_list",
            "embeddings",
            "total_labels",
            "recon_loss",
        ]
    
    @property
    def input_list(self):
        return ["features", "labels"]
    
    @property
    def _default_next_module(self):
        return "collector"
    
    def get_params(self):
        name_list = ["node_{}"]
        # get embedders
        for i in range(self.embedder_group):
            for meta_name in name_list:
                setattr(
                    self,
                    meta_name.format(i),
                    nn.Linear(self.input_size, self.embedding_size)
                )
            setattr(
                self,
                "decoder_{}".format(i),
                nn.Linear(self.embedding_size, 1024)
            )
    
    def initiate_params(self):
        name_list = ["node_{}"]
        # initiate embedders
        for i in range(self.embedder_group):
            for meta_name in name_list:
                embedder = getattr(self, meta_name.format(i))
                nn.init.kaiming_normal_(embedder.weight.data, mode="fan_out")
                nn.init.constant_(embedder.bias.data, 0)
            decoder = getattr(self, "decoder_{}".format(i))
            nn.init.kaiming_normal_(decoder.weight.data, mode="fan_out")
            nn.init.constant_(decoder.bias.data, 0)

    
    def recon_func(self, ori_feat_list, recon_feat_list):
        """
        Compute the reconstruction loss of N branch for one sample
        and return the smallest index of the branch.

        Args:
            ori_feat_list (list):
                Size: batch-size x feat-dim
            recon_feat_list (list):
                Format: {index: torch.Tensor}
        
        Returns:
            torch.Tensor: reconstruction loss for each branch
            Size: batch-size x branch-num
        """
        # compute the recon-loss
        recon_loss = []
        for i in range(self.embedder_group):
            recon_loss.append(F.mse_loss(
                ori_feat_list, recon_feat_list[i], reduction="none"
            ).mean(dim=1))
        recon_loss = torch.stack(recon_loss, dim=1)

        min_index = torch.min(recon_loss, dim=1)[1]

        index_dict = {}
        for i in range(self.embedder_group):
            index_dict[i] = torch.where(min_index == i)[0]

        # compute the final recon_loss
        mean_recon_loss = torch.mean(recon_loss)
        return mean_recon_loss, index_dict
    
    def _forward_func(self, features, labels=None):
        """
        Use multi-encoder and multi-decoder to select features

        if labels is None, ``test`` mode is activated!
        """
        features_detach = features.detach()
        # split features
        split_features = list(torch.chunk(features, self.embedder_group, dim=-1))
        split_features_detach = [
            sub_feature.detach()
            for sub_feature in split_features
        ]

        # compute nodes
        self.nodes_list = []
        self.detached_nodes_list = []
        for i in range(self.embedder_group):
            node_f = getattr(self, "node_{}".format(i))
            cur_node = node_f(split_features[i])
            # cur_node = torch.nn.functional.normalize(
            #     cur_node, dim=-1, p=2
            # )
            self.nodes_list.append(cur_node)
            self.detached_nodes_list.append(cur_node.detach())
        
        nodes_updated_final = torch.cat(self.nodes_list, dim=1)
        
        # compute the reconstruction features
        self.recon_feat_list = []
        for i in range(self.embedder_group):
            decoder = getattr(self, "decoder_{}".format(i))
            # detach to stop the gradients
            self.recon_feat_list.append(decoder(
                self.detached_nodes_list[i]
            ))
        
        # compute the reconstruction loss
        self.recon_loss, self.index_dict = self.recon_func(features_detach, self.recon_feat_list)


        # split by index-dict
        if labels is not None: # for training
            self.labels_list = []
            for i in range(self.embedder_group):
                self.nodes_list[i] = self.nodes_list[i][self.index_dict[i]]
                self.labels_list.append(labels[self.index_dict[i]])
            return (
                self.nodes_list,
                self.labels_list,
                nodes_updated_final,
                labels,
                self.recon_loss,
                # self.div_loss,
            )
        else: # for testing
            return nodes_updated_final

    def output_wrapper(self, output_tuple: tuple, *args, **kwargs) -> dict:
        (
            nodes_list,
            labels_list,
            embeddings,
            total_labels,
            recon_loss,
            # div_loss,
        ) = output_tuple
        output_dict = {
            "collectors/ensemble_proxy/": {
                "embeddings": nodes_list,
                "labels": labels_list
            },
            "collectors/repre_proxy/loss_proxy": {
                "embeddings": embeddings,
                "labels": total_labels,
            },
            "FINISH/default/loss_recon": {
                "loss": recon_loss,
                "weight": 0.1
            },
        }
        return output_dict
    
    # def set_stats(self):
    #     # NOTE: statistics of index number
    #     for i in range(self.embedder_group):
    #         setattr(
    #             self,
    #             "num_branch_{}".format(i),
    #             len(self.index_dict[i])
    #         )