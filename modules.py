import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max
from utils import build_mlps, gumbel_softmax


class HypergraphConstruction(nn.Module):

    """Hypergraph Construction module as proposed in our paper"""

    def __init__(self, config):
        super(HypergraphConstruction, self).__init__()
        self._config = config
        self._agent_encoder = TypeSpecificEncoder(config["type_specific_encoder_config"])
        self._polyline_encoder = PolylineEncoder(config["polyline_encoder_config"])
        self._scales = config["scales"]

    def init_Hyperedges(self, agent_features, agent_features_corr, scale=2):
        batch = agent_features.shape[0]
        agent_number = agent_features.shape[1]

        if scale == agent_number:
            H_matrix = torch.ones(batch,1,agent_number).type_as(agent_features)
            return H_matrix
        group_size = scale
        if group_size < 1:
            group_size = 1
        if group_size > agent_number:
            group_size = agent_number
        
        _,indice = torch.topk(agent_features_corr,dim=2,k=group_size,largest=True)
        H_matrix = torch.zeros(batch,agent_number,agent_number).type_as(agent_features)
        H_matrix = H_matrix.scatter(dim=2,index=indice,src=torch.ones_like(indice).type_as(agent_features))

        return H_matrix
    def forward(self, batch_dict):
        self._agent_types = batch_dict['obj_types']
        polyline_features = self._polyline_encoder.forward(polylines=batch_dict['map_polylines'], polylines_mask=batch_dict['map_polylines_mask'])
        agent_features = self._agent_encoder.forward(scatter_numbers=batch_dict['scatter_numbers'], scatter_idx=batch_dict['scatter_idx'], 
                                                     lstm_data=batch_dict['lstm_data'], lstm_data_diff=batch_dict['lstm_data_diff'], 
                                                     mcg_data=batch_dict['mcg_data'], agent_type=batch_dict['obj_types'])
        query = F.normalize(agent_features, dim=-1)
        key = F.normalize(agent_features, dim=-1)
        corr = torch.matmul(query, key.transpose(-1, -2))

        H_matrices = []
        for scale in self._scales:
            H_matrices.append(self.init_Hyperedges(agent_features, corr, scale=scale))
        H_matrices = torch.cat(H_matrices, dim=1)

        return agent_features, polyline_features, H_matrices

class TATLMPLayer(nn.Module):
    """Type-Aware Two-Level Message Passing (TATLMP) module as proposed in our paper"""

    def __init__(self, scale, config):
        super(TATLMPLayer, self).__init__()
        self._config = config
        self._weight_mlp = build_mlps(**self._config['weight_mlp'])
        self._output_mlp = build_mlps(**self._config['output_mlp'])
        self._distribution_mlp = build_mlps(**self._config['distribution_mlp'])
        self._factor_mlp = build_mlps(**self._config['factor_mlp'])
        self._MHA = MHA(self._config['MHA'])
        self._edge_types = self._config['edge_types']
        self._scale = scale
        self._learned_type_embeddings = nn.Parameter(torch.zeros([self._edge_types ,self._config['h_dim']]))
        # initialize the learned type embeddings
        nn.init.xavier_uniform_(self._learned_type_embeddings)
    def forward(self, agent_features, agent_types, polyline_features, H):
        """
        Args:
            agent_features (batch_size, num_agents, D):
            agent_types (batch_size, num_agents, total_types):
            polylines_mask (batch_size, num_polylines, D):
            H (batch_size, num_edge, num_agents):
        Returns:
            encoded polyline features (batch_size, num_polylines, D):
         Constraints:
        - the map between edge type and edge typing embedding is not efficient
        """
        # Intra-group Aggregation 
        batch = agent_features.shape[0]
        agent_nubmer = agent_features.shape[1]
        edge_number = H.shape[1]
        edge_features_init = torch.matmul(H,agent_features)
        agent_features = (agent_features[:,:,None,:].transpose(2,1)).repeat(1,edge_number,1,1)
        edge_features_init_rep = edge_features_init[:,:,None,:].repeat(1,1,agent_nubmer,1)
        edges_weight = self._weight_mlp(torch.cat((edge_features_init_rep, agent_features),dim=-1))[:,:,:,0]
        H_weighted = edges_weight * H
        H_weighted = F.softmax(H_weighted,dim=2)
        H_weighted = H_weighted * H
        edge_features = torch.matmul(H_weighted, agent_features)
        # Inter-group Aggregation
        edge_type= torch.matmul(H, agent_types) # (B, E, T)
        distribution = gumbel_softmax(self._distribution_mlp(torch.concatenate([edge_features, edge_type])),tau=1/2, hard=False) # (B, E, T)
        factor = torch.sigmoid(self._factor_mlp(torch.concatenate([edge_features, edge_type]))) # (B, E, 1)
        edge_type_embedding = factor * distribution * self._learned_type_embeddings # (B, E, D)
        edge_type_feature_embedding = edge_type_feature_embedding[:,:, None,:].repeat(1, 1, agent_nubmer, 1)
        agent_features_rep = agent_features[:, None, :, :].repeat(1, edge_number, 1, 1)
        H_mask = H[..., None]
        agents_related_edge_feature = torch.cat((edge_type_feature_embedding, agent_features_rep), dim=-1) 
        agents_related_edge_feature = agents_related_edge_feature * H_mask
        H_weighted = F.leaky_relu(self.weight_mlp(agents_related_edge_feature).squeeze(-1))
        H_weighted = F.softmax(H_weighted,dim=2)
        H_weighted = H_weighted * H
        # Scene Context Fusion
        agent_features = torch.cat((torch.matmul(H_weighted.permute(0,2,1), torch.cat((edge_features, edge_type_embedding), dim=-1)),agent_features),dim=-1) 
        polyline_features_new = self._MHA(agent_features, polyline_features)
        agent_features_new = self._output_mlp(torch.concatenate([agent_features, polyline_features_new], dim=-1))

        return agent_features_new, polyline_features_new


class PolylineEncoder(nn.Module):
    def __init__(self, config):
        super(PolylineEncoder, self).__init__()
        self._config = config
        in_channel = self._config["in_channel"]
        hidden_dim = self._config["hidden_dim"]
        num_layers = self._config["num_layers"]
        num_pre_layers = self._config["num_pre_layers"]
        out_channel = self._config["out_channel"]
        self._pre_mlps = build_mlps(
            c_in=in_channel,
            mlp_channels=[hidden_dim] * num_pre_layers, 
            activation='relu', 
            ret_before_act=False, 
            without_norm=False
            )
        self._mlps = build_mlps(
            c_in=hidden_dim * 2,
            mlp_channels=[hidden_dim] * (num_layers - num_pre_layers),
            ret_before_act=False
        )
        self._out_mlp = build_mlps(
            c_in=hidden_dim,
            mlp_channels=[out_channel],
            ret_before_act=True
        )
    def forward(self, polylines, polylines_mask):
        """
        Args:
            polylines (batch_size, num_polylines, num_points_each_polylines, C):
            polylines_mask (batch_size, num_polylines, num_points_each_polylines):
        Returns:
            encoded polyline features (batch_size, num_polylines, D):
        """
        batch_size, num_polylines,  num_points_each_polylines, C = polylines.shape  
        # encode the valid polyline points
        polylines_feature_valid = self._pre_mlps(polylines[polylines_mask])
        polylines_feature = polylines.new_zeros(batch_size, num_polylines,  num_points_each_polylines, polylines_feature_valid.shape[-1])
        polylines_feature[polylines_mask] = polylines_feature_valid      
        
        pooled_feature = polylines_feature.max(dim=2)[0]
        polylines_feature = torch.cat((polylines_feature, pooled_feature[:, :, None, :].repeat(1, 1, num_points_each_polylines, 1)), dim=-1)  

        polylines_feature_valid = self._mlps(polylines_feature[polylines_mask])
        polyline_features = polylines_feature.new_zeros(batch_size, num_polylines, num_points_each_polylines, polylines_feature_valid.shape[-1])
        polyline_features[polylines_mask] = polylines_feature_valid
        
        polyline_features = polyline_features.max(dim=2)[0]  # (batch_size, num_polylines, D)

        valid_mask = (polylines_mask.sum(dim=-1) > 0)
        polyline_features_valid = self.out_mlps(polyline_features[valid_mask])  # (N, D)
        polyline_features = polyline_features.new_zeros(batch_size, num_polylines, polyline_features.shape[-1])
        polyline_features[valid_mask] = polyline_features_valid

        return polyline_features

class TypeSpecificEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self._config = config
        self._agent_type = config["agent_type"]

        self._lstm = nn.ModuleList()
        self._diff_lstm = nn.ModuleList()

        for _ in range(self._agent_type):
            self._lstm.append(nn.LSTM(batch_first=True, **config["lstm_config"]))
            self._diff_lstm.append(nn.LSTM(batch_first=True, **config["diff_lstm_config"]))

        self._mcg = MCGBlock(config["mcg_config"])
        self._lstm_hidden_dim = config["position_lstm_config"]["hidden_dim"]
        self._diff_lstm_hidden_dim = config["position_diff_lstm_config"]["hidden_dim"]

    def forward(self, scatter_numbers, scatter_idx, lstm_data, lstm_data_diff, mcg_data, agent_type):
        
        batch_size = scatter_idx.size(0)
        lstm_embedding = torch.zeros(batch_size, 1, self._lstm_hidden_dim).cuda()
        diff_lstm_embedding = torch.zeros(batch_size, 1, self._diff_lstm_hidden_dim).cuda()
        
        for agent_type_idx in range(self._agent_type):
            idx = torch.where(agent_type == agent_type_idx)[0]
            if idx.size(0) == 0: continue
            lstm_embedding[idx, :, :] = self._lstm[agent_type_idx](lstm_data[idx])[0][:, -1, :]
            diff_lstm_embedding[idx, :, :] = self._diff_lstm[agent_type_idx](lstm_data_diff[idx])[0][:, -1, :] 
        
        mcg_embedding = self._mcg(scatter_numbers, scatter_idx, mcg_data, aggregate_batch=False)
       
        return torch.cat([
                            lstm_embedding, 
                            diff_lstm_embedding, 
                            mcg_embedding], 
                            axis=-1
                        )
class MHA(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_in = config['n_in'] 
        n_out = config['n_out']
        self._config = config
        self._q = nn.Linear(n_in, n_out)
        self._k = nn.Linear(n_in, n_out)
        self._v = nn.Linear(n_in, n_out)
        self._mha = nn.MultiheadAttention(n_out, 4, batch_first=True)
    
    def forward(self, q, k_v):
        batch_size = q.shape[0]
        Q = self._q(q)
        K = self._k(k_v)
        V = self._v(k_v)
        out, _ = self._mha(Q, K, V)
        return out

class CGBlock(nn.Module):
    """
    NOTE: 
    based on https://github.com/sshaoshuai/MTR
    (Apache License)
    """
    def __init__(self, config):
        super().__init__()
        self._config = config
        self.s_mlp = build_mlps(**config["mlp"])
        self.c_mlp = nn.Identity() if config["identity_c_mlp"] else build_mlps(**config["mlp"])
        self.n_in = self.s_mlp.n_in
        self.n_out = self.s_mlp.n_out

    def forward(self, scatter_numbers, s, c):
        prev_s_shape, prev_c_shape = s.shape, c.shape
        s = self.s_mlp(s.view(-1, s.shape[-1])).view(prev_s_shape)
        c = self.c_mlp(c.view(-1, c.shape[-1])).view(prev_c_shape)
        s = s * c
        if self._config["agg_mode"] == "max":
            aggregated_c = torch.max(s, dim=1, keepdim=True)[0]
        elif self._config["agg_mode"] in ["mean", "avg"]:
            aggregated_c = torch.mean(s, dim=1, keepdim=True)
        else:
            raise Exception("Unknown agg mode for MCG")
        return s, aggregated_c

class MCGBlock(nn.Module):
    """
    NOTE: 
    based on https://github.com/sshaoshuai/MTR
    (Apache License)
    """
    def __init__(self, config):
        super().__init__()
        self._config = config
        self._blocks = []
        for i in range(config["n_blocks"]):
            current_block_config = config["block"].copy()
            if i == 0 and config["identity_c_mlp"]:
                current_block_config["identity_c_mlp"] = True
            else:
                current_block_config["identity_c_mlp"] = False
            current_block_config["agg_mode"] = config["agg_mode"]
            self._blocks.append(CGBlock(current_block_config))
        self._blocks = nn.ModuleList(self._blocks)
        self.n_in = self._blocks[0].n_in
        self.n_out = self._blocks[-1].n_out
    
    def _repeat_tensor(self, tensor, scatter_numbers, axis=0):
        result = []
        for i in range(len(scatter_numbers)):
            result.append(tensor[[i]].expand((int(scatter_numbers[i]), -1, -1)))
        result = torch.cat(result, axis=0)
        return result

    def _compute_running_mean(self, prevoius_mean, new_value, i):
        if self._config["running_mean_mode"] == "real":
            result = (prevoius_mean * i + new_value) / i
        elif self._config["running_mean_mode"] == "sliding":
            assert self._config["alpha"] + self._config["beta"] == 1
            result = self._config["alpha"] * prevoius_mean + self._config["beta"] * new_value
        return result

    def forward(
            self, scatter_numbers, scatter_idx, s, c=None, aggregate_batch=True, return_s=False):
        if c is None:
            assert self._config["identity_c_mlp"], self._config["identity_c_mlp"]
            c = torch.ones(s.shape[0], 1, self.n_in, requires_grad=True).cuda()
        else:
            assert not self._config["identity_c_mlp"]
        c = self._repeat_tensor(c, scatter_numbers)
        assert torch.isfinite(s).all()
        assert torch.isfinite(c).all()
        running_mean_s, running_mean_c = s, c
        for i, cg_block in enumerate(self._blocks, start=1):
            s, c = cg_block(scatter_numbers, running_mean_s, running_mean_c)
            assert torch.isfinite(s).all()
            assert torch.isfinite(c).all()
            running_mean_s = self._compute_running_mean(running_mean_s, s, i)
            running_mean_c = self._compute_running_mean(running_mean_c, c, i)
            assert torch.isfinite(running_mean_s).all()
            assert torch.isfinite(running_mean_c).all()
        if return_s:
            return running_mean_s 
        if aggregate_batch:
            return scatter_max(running_mean_c, scatter_idx, dim=0)[0]
        return running_mean_c

