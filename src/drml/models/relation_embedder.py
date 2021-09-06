import torch
import torch.nn as nn 
import torch.nn.functional as F

from thudml.core.modules import WithRecorder

class RelationEmbedder(WithRecorder):
    def __init__(
        self,
        embedding_size=128,
        embedder_group=1,
        input_size=256,
        div_gamma=1,
        weight_repre_loss=1,
        weight_recon_loss=1,
    ):
        super(RelationEmbedder, self).__init__()

        self.embedding_size = embedding_size
        self.embedder_group = embedder_group
        self.input_size = input_size
        self.div_gamma = div_gamma
        self.weight_repre_loss = weight_repre_loss
        self.weight_recon_loss = weight_recon_loss

        self.get_params()
        self.initiate_params()

        to_record_list = [
            "num_branch_{}".format(i)
            for i in range(self.embedder_group)
        ]
        for item in to_record_list:
            self.add_recordable_attr(name=item)
        
    def get_params(self):
        name_list = ["node_{}", "Aedge_{}", "Bedge_{}"]
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
                nn.Linear(self.embedding_size, self.input_size)
            )
        # score function
        self.score_func = nn.Linear(self.embedding_size, 1)
        # integrator
        self.integrator = nn.Linear(
            self.embedding_size * 2,
            self.embedding_size
        )
    
    def initiate_params(self):
        name_list = ["node_{}", "Aedge_{}", "Bedge_{}"]
        # initiate embedders
        for i in range(self.embedder_group):
            for meta_name in name_list:
                embedder = getattr(self, meta_name.format(i))
                nn.init.kaiming_normal_(embedder.weight.data, mode="fan_out")
                nn.init.constant_(embedder.bias.data, 0)
            decoder = getattr(self, "decoder_{}".format(i))
            nn.init.kaiming_normal_(decoder.weight.data, mode="fan_out")
            nn.init.constant_(decoder.bias.data, 0)
        # score function
        nn.init.kaiming_normal_(self.score_func.weight.data, mode="fan_out")
        nn.init.constant_(self.score_func.bias.data, 0)
        # integrator
        # nn.init.kaiming_normal_(self.integrator.weight.data, mode="fan_out")
        self.integrator.weight.data = torch.cat(
            [torch.eye(self.embedding_size), torch.zeros(self.embedding_size, self.embedding_size)],
            dim=1
        )
        nn.init.constant_(self.integrator.bias.data, 0)
    
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
                ori_feat_list[i], recon_feat_list[i], reduction="none"
            ).mean(dim=1))
        recon_loss = torch.stack(recon_loss, dim=1)
        min_index = torch.min(recon_loss, dim=1)[1]
        index_dict = {}
        for idx in range(self.embedder_group):
            index_dict[idx] = min_index == idx

        # compute the final recon_loss
        mean_recon_loss = torch.mean(recon_loss)
        return mean_recon_loss, index_dict
    
    def forward(self, features, labels=None):
        """
        Use multi-encoder and multi-decoder to select features

        if labels is None, ``test`` mode is activated!
        """
        # split features
        features_detach = features.detach()
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
            cur_node = torch.nn.functional.normalize(
                cur_node, dim=-1, p=2
            ) # normalize each branch
            self.nodes_list.append(cur_node)
            self.detached_nodes_list.append(cur_node.detach())
        
        # compute edges
        self.Aedges_list, self.Bedges_list = [], []
        for i in range(self.embedder_group):
            Aedge_f = getattr(self, "Aedge_{}".format(i))
            Bedge_f = getattr(self, "Bedge_{}".format(i))
            self.Aedges_list.append(Aedge_f(split_features_detach[i]))
            self.Bedges_list.append(Bedge_f(split_features_detach[i]))
        
        # compute the reconstruction features
        self.recon_feat_list = []
        for i in range(self.embedder_group):
            decoder = getattr(self, "decoder_{}".format(i))
            # detach to stop the gradients
            self.recon_feat_list.append(decoder(
                self.detached_nodes_list[i]
            ))
        
        # compute the reconstruction loss
        self.recon_loss, self.index_dict = self.recon_func(split_features_detach, self.recon_feat_list)

        # compute relational updated nodes
        nodes = torch.stack(self.nodes_list, dim=2) #.detach()
        Aedges = torch.stack(self.Aedges_list, dim=1)
        Bedges = torch.stack(self.Bedges_list, dim=1)
        
        diff_edges = Aedges.unsqueeze(2) - Bedges.unsqueeze(1)
        score_edges = self.score_func(diff_edges).squeeze()
        score_edges = torch.nn.functional.relu(score_edges)
        normed_score_edges = torch.nn.functional.normalize(score_edges, p=1, dim=1)

        nodes_updated = torch.matmul(nodes, normed_score_edges)
        nodes_updated_cat = torch.cat([nodes, nodes_updated], dim=1).permute(0, 2, 1)
        bs = nodes_updated_cat.size(0)
        nodes_updated_final = self.integrator(nodes_updated_cat).view(bs, -1)

        # set statistics
        self.set_stats()

        # split by index-dict
        if self.training:
            self.labels_list = []
            for idx in range(self.embedder_group):
                self.labels_list.append(labels[self.index_dict[idx]])
                self.nodes_list[idx] = self.nodes_list[idx][self.index_dict[idx]]
            return (
                self.nodes_list,
                self.labels_list,
                nodes_updated_final,
                labels,
                self.weight_repre_loss,
                self.recon_loss,
                self.weight_recon_loss,
            )
        else:
            return nodes_updated_final
    
    def set_stats(self):
        # NOTE: statistics of index number
        for i in range(self.embedder_group):
            setattr(
                self,
                "num_branch_{}".format(i),
                sum(self.index_dict[i])
            )