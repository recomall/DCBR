import math
import numpy as np
import scipy.sparse as sp 
import torch
import torch.nn as nn
import torch.nn.functional as F


class DCBR(nn.Module):
    def __init__(self, conf, raw_graph):
        super().__init__()
        self.conf = conf
        self.device = conf["device"]

        self.embedding_size = conf["embedding_size"]
        self.num_users = conf["num_users"]
        self.num_bundles = conf["num_bundles"]
        self.num_items = conf["num_items"]
        
        self.num_layers = conf["num_layers"]
        self.gamma_1 = conf["gamma_1"]
        self.gamma_2 = conf["gamma_2"]
        self.tau = conf["tau"]
        self.lambda_1 = conf["lambda_1"]
        self.lambda_2 = conf["lambda_2"]

        self.ub_graph, self.ui_graph, self.bi_graph = raw_graph
        self.UI_propagation_graph = self.get_propagation_graph(self.ui_graph)
        self.UI_aggregation_graph = self.get_aggregation_graph(self.ui_graph)
        self.BI_propagation_graph = self.get_propagation_graph(self.bi_graph)
        self.BI_aggregation_graph = self.get_aggregation_graph(self.bi_graph)

        self.upsilon_dict = {"UB": conf["upsilon_UB"], "UI": conf["upsilon_UI"], "BI": conf["upsilon_BI"]}
        self.modal_coefs = torch.FloatTensor([conf["omega"], 1 - conf["omega"]]).unsqueeze(-1).unsqueeze(-1).to(self.device)
        self.UB_layer_coefs = torch.FloatTensor(conf["xi_UB"]).unsqueeze(0).unsqueeze(-1).to(self.device)
        self.UI_layer_coefs = torch.FloatTensor(conf["xi_UI"]).unsqueeze(0).unsqueeze(-1).to(self.device)
        self.BI_layer_coefs = torch.FloatTensor(conf["xi_BI"]).unsqueeze(0).unsqueeze(-1).to(self.device)
        
        self.users_feature = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
        nn.init.xavier_normal_(self.users_feature)
        self.bundles_feature = nn.Parameter(torch.FloatTensor(self.num_bundles, self.embedding_size))
        nn.init.xavier_normal_(self.bundles_feature)
        self.items_feature = nn.Parameter(torch.FloatTensor(self.num_items, self.embedding_size))
        nn.init.xavier_normal_(self.items_feature)
        
        self.restore_user_e = None
        self.restore_bundle_e = None
        self.other_params_name = ["restore_user_e", "restore_bundle_e"]


    def other_params(self):
        if hasattr(self, "other_params_name"):
            return {key: getattr(self, key) for key in self.other_params_name}
        return dict()


    def getUserEmbeds(self):
        return self.users_feature


    def getBundleEmbeds(self):
        return self.bundles_feature


    def get_propagation_graph(self, bipartite_graph):
        propagation_graph = sp.bmat([[sp.csr_matrix((bipartite_graph.shape[0], bipartite_graph.shape[0])), bipartite_graph], [bipartite_graph.T, sp.csr_matrix((bipartite_graph.shape[1], bipartite_graph.shape[1]))]])
        rowsum_sqrt = sp.diags(1/(np.sqrt(propagation_graph.sum(axis=1).A.ravel()) + 1e-8))
        colsum_sqrt = sp.diags(1/(np.sqrt(propagation_graph.sum(axis=0).A.ravel()) + 1e-8))
        propagation_graph = rowsum_sqrt @ propagation_graph @ colsum_sqrt
        propagation_graph = propagation_graph.tocoo()
        values = propagation_graph.data
        indices = np.vstack((propagation_graph.row, propagation_graph.col))
        propagation_graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(propagation_graph.shape)).to(self.device)
        return propagation_graph


    def get_aggregation_graph(self, bipartite_graph):
        bundle_size = bipartite_graph.sum(axis=1) + 1e-8
        bipartite_graph = sp.diags(1/bundle_size.A.ravel()) @ bipartite_graph
        bipartite_graph = bipartite_graph.tocoo()
        values = bipartite_graph.data
        indices = np.vstack((bipartite_graph.row, bipartite_graph.col))
        bipartite_graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(bipartite_graph.shape)).to(self.device)
        return bipartite_graph


    def graph_propagate(self, graph, A_feature, B_feature, graph_type, layer_coef, test):
        features = torch.cat((A_feature, B_feature), dim=0)
        all_features = [features]
        for _ in range(self.num_layers):
            features = torch.spmm(graph, features)
            if not test:
                random_noise = torch.rand_like(features).to(self.device)
                features += torch.sign(features) * F.normalize(random_noise, dim=-1) * self.upsilon_dict[graph_type]
            all_features.append(F.normalize(features, p=2, dim=1))
        all_features = torch.stack(all_features, dim=1) * layer_coef
        all_features = torch.sum(all_features, dim=1)
        A_feature, B_feature = torch.split(all_features, (A_feature.shape[0], B_feature.shape[0]), dim=0)
        return A_feature, B_feature


    def graph_aggregate(self, agg_graph, node_feature, graph_type, test):
        aggregated_feature = torch.matmul(agg_graph, node_feature)
        if not test:
            random_noise = torch.rand_like(aggregated_feature).to(self.device)
            aggregated_feature += torch.sign(aggregated_feature) * F.normalize(random_noise, dim=-1) * self.upsilon_dict[graph_type]
        return aggregated_feature


    def propagate(self, UB_propagation_graph=None, test=False):
        if UB_propagation_graph is not None and not test:
            UB_users_feature, UB_bundles_feature = self.graph_propagate(UB_propagation_graph, self.users_feature, self.bundles_feature, "UB", self.UB_layer_coefs, test)
        UI_users_feature, UI_items_feature = self.graph_propagate(self.UI_propagation_graph, self.users_feature, self.items_feature, "UI", self.UI_layer_coefs, test)
        UI_bundles_feature = self.graph_aggregate(self.BI_aggregation_graph, UI_items_feature, "BI", test)
        BI_bundles_feature, BI_items_feature = self.graph_propagate(self.BI_propagation_graph, self.bundles_feature, self.items_feature, "BI", self.BI_layer_coefs, test)
        BI_users_feature = self.graph_aggregate(self.UI_aggregation_graph, BI_items_feature, "UI", test)

        users_rep = torch.sum(torch.stack([UI_users_feature, BI_users_feature], dim=0) * self.modal_coefs, dim=0)
        bundles_rep = torch.sum(torch.stack([UI_bundles_feature, BI_bundles_feature], dim=0) * self.modal_coefs, dim=0)

        if test:
            return users_rep, bundles_rep
        else:
            users_feature = [UB_users_feature, UI_users_feature, BI_users_feature]
            bundles_feature = [UB_bundles_feature, UI_bundles_feature, BI_bundles_feature]
            return users_rep, bundles_rep, users_feature, bundles_feature


    def cal_reg_loss(self):
        reg_loss = 0
        for W in self.parameters():
            reg_loss += W.norm(2).square()
        return reg_loss

 
    def cal_bpr_loss(self, users_feature, bundles_feature):
        pred = torch.sum(users_feature * bundles_feature, dim=2)
        if pred.shape[1] > 2:
            negs = pred[:, 1:]
            pos = pred[:, 0].unsqueeze(1).expand_as(negs)
        else:
            negs = pred[:, 1].unsqueeze(1)
            pos = pred[:, 0].unsqueeze(1)
        bpr_loss = - torch.mean(torch.log(torch.sigmoid(pos - negs)))
        return bpr_loss + self.lambda_2 * self.cal_reg_loss()


    def cal_cl_loss(self, pos, aug):
        pos = F.normalize(pos, p=2, dim=1)
        aug = F.normalize(aug, p=2, dim=1)
        pos_score = torch.sum(pos * aug, dim=1)
        pos_score = torch.exp(pos_score / self.tau)
        ttl_score = torch.matmul(pos, aug.transpose(0, 1))
        ttl_score = torch.sum(torch.exp(ttl_score / self.tau), dim=1)
        cl_loss = - torch.mean(torch.log(pos_score / ttl_score))
        return cl_loss


    def forward(self, UB_propagation_graph, batch):
        users, bundles = batch
        users_rep, bundles_rep, users_feature, bundles_feature = self.propagate(UB_propagation_graph)

        users_embedding = users_rep[users].expand(-1, bundles.shape[1], -1)
        bundles_embedding = bundles_rep[bundles]
        bpr_loss = self.cal_bpr_loss(users_embedding, bundles_embedding)

        user = users[:, 0]
        bundle = bundles[:, 0]
        ub_users_embedding = users_feature[0][user]
        ub_bundles_embedding = bundles_feature[0][bundle]
        ui_users_embedding = users_feature[1][user]
        ui_bundles_embedding = bundles_feature[1][bundle]
        bi_users_embedding = users_feature[2][user]
        bi_bundles_embedding = bundles_feature[2][bundle]

        u_cl_inter = self.cal_cl_loss(ub_users_embedding, ui_users_embedding) \
                   + self.cal_cl_loss(ub_users_embedding, bi_users_embedding) \
                   + self.cal_cl_loss(ui_users_embedding, bi_users_embedding)
        
        b_cl_inter = self.cal_cl_loss(ub_bundles_embedding, ui_bundles_embedding) \
                   + self.cal_cl_loss(ub_bundles_embedding, bi_bundles_embedding) \
                   + self.cal_cl_loss(ui_bundles_embedding, bi_bundles_embedding)
        
        u_cl_intra = self.cal_cl_loss(ub_users_embedding, ub_users_embedding) \
                   + self.cal_cl_loss(ui_users_embedding, ui_users_embedding) \
                   + self.cal_cl_loss(bi_users_embedding, bi_users_embedding) \

        b_cl_intra = self.cal_cl_loss(ub_bundles_embedding, ub_bundles_embedding) \
                   + self.cal_cl_loss(ui_bundles_embedding, ui_bundles_embedding) \
                   + self.cal_cl_loss(bi_bundles_embedding, bi_bundles_embedding)
        
        cl_loss = self.gamma_1 * (u_cl_inter + b_cl_inter) + self.gamma_2 * (u_cl_intra + b_cl_intra)
        cl_loss *= self.lambda_1

        return bpr_loss, cl_loss


    def evaluate(self, propagate_result, users):
        self.restore_user_e, self.restore_bundle_e = propagate_result
        scores = torch.matmul(self.restore_user_e[users], self.restore_bundle_e.transpose(0, 1))
        return scores


class DNN(nn.Module):
    def __init__(self, in_dims, out_dims, time_emb_dim, norm=False, dropout=0.5):
        super(DNN, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.time_emb_dim = time_emb_dim
        self.norm = norm
        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
        in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
        out_dims_temp = self.out_dims
        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
        self.drop = nn.Dropout(dropout)
        self.init_weights()


    def init_weights(self):
        for layer in self.in_layers:
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)
        for layer in self.out_layers:
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)
        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)


    def forward(self, x, timesteps, use_dropout=True, max_period=10000):
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=self.time_emb_dim // 2, dtype=torch.float32) / (self.time_emb_dim // 2)).cuda()
        args = timesteps[:, None].float() * freqs[None]
        timestep_embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.time_emb_dim % 2:
            timestep_embedding = torch.cat([timestep_embedding, torch.zeros_like(timestep_embedding[:, :1])], dim=-1)
        emb = self.emb_layer(timestep_embedding)
        if self.norm:
            x = F.normalize(x)
        if use_dropout:
            x = self.drop(x)
        h = torch.cat([x, emb], dim=-1)
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)
        return h


class GaussianDiffusion(nn.Module):
    def __init__(self, noise_scale, noise_min, noise_max, steps, beta_fixed=True):
        super(GaussianDiffusion, self).__init__()
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps
        if noise_scale != 0:
            self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).cuda()
            if beta_fixed:
                self.betas[0] = 0.0001
            self.calculate_for_diffusion()


    def get_betas(self, max_beta=0.999):
        start = self.noise_scale * self.noise_min
        end = self.noise_scale * self.noise_max
        variance = np.linspace(start, end, self.steps, dtype=np.float64)
        alpha_bar = 1 - variance
        betas = []
        betas.append(1 - alpha_bar[0])
        for i in range(1, self.steps):
            betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
        return np.array(betas) 


    def calculate_for_diffusion(self):
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, axis=0).cuda()
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).cuda(), self.alphas_cumprod[:-1]]).cuda()
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).cuda()]).cuda()

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod)


    def p_sample(self, model, x_start, steps, sampling_noise=False):
        if steps == 0:
            x_t = x_start
        else:
            t = torch.tensor([steps - 1] * x_start.shape[0]).cuda()
            x_t = self.q_sample(x_start, t)
        
        indices = list(range(self.steps))[::-1]

        if self.noise_scale == 0.:
            for i in indices:
                t = torch.tensor([i] * x_t.shape[0]).to(x_start.device)
                x_t = model(x_t, t)
            return x_t
        
        for i in indices:
            t = torch.tensor([i] * x_t.shape[0]).cuda()
            out = self.p_mean_variance(model, x_t, t)
            if sampling_noise:
                noise = torch.randn_like(x_t)
                nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
                x_t = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
            else:
                x_t = out["mean"]
        return x_t


    def training_CBDM_losses(self, model, x_start, U_Embeds, B_Embeds, batch_user_index):
        batch_size = x_start.size(0)
        ts = torch.randint(0, self.steps, (batch_size,)).long().cuda()
        noise = torch.randn_like(x_start)
        if self.noise_scale != 0:
            x_t = self.q_sample(x_start, ts, noise)
        else:
            x_t = x_start
        
        model_output = model(x_t, ts)
        mse = self.mean_flat((x_start - model_output) ** 2)
        weight = self.SNR(ts - 1) - self.SNR(ts)
        weight = torch.where((ts == 0), 1.0, weight)
        elbo_loss = weight * mse

        B_new_embeds = torch.mm(model_output.transpose(0, 1), U_Embeds[batch_user_index])
        blcc_loss = self.mean_flat((B_new_embeds - B_Embeds) ** 2)

        return elbo_loss.mean(), blcc_loss.mean()


    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise


    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start + self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        return posterior_mean


    def p_mean_variance(self, model, x, t):
        model_output = model(x, t, False)
        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped
        model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)
        model_mean = self.q_posterior_mean_variance(x_start=model_output, x_t=x, t=t)
        return {
            "mean": model_mean,
            "log_variance": model_log_variance
        }


    def SNR(self, t):
        self.alphas_cumprod = self.alphas_cumprod.cuda()
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])


    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        arr = arr.cuda()
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)


    def mean_flat(self, tensor):
        return tensor.mean(dim=list(range(1, len(tensor.shape))))