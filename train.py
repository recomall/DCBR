import os
import yaml
import random
import argparse
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
import torch
from utility import Datasets, DiffusionDataset, write_log
from models.DCBR import DCBR, DNN, GaussianDiffusion


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", type=str, default="0")
    parser.add_argument("-d", "--dataset", type=str, default="iFashion")
    parser.add_argument("-m", "--model", type=str, default="DCBR")
    parser.add_argument("-i", "--info", type=str, default="")
    parser.add_argument("-s", "--seed", type=int, default=0)
    args = parser.parse_args()
    return args


def main():
    paras = get_cmd().__dict__
    set_seed(paras["seed"])
    conf_overall = yaml.safe_load(open("configs/overall.yaml"))
    conf_model = yaml.safe_load(open(f"configs/models/{paras['model']}.yaml"))
    print("load config file done!")

    dataset_name = paras["dataset"]
    assert paras["model"] in ["DCBR"], "Pls select models from: DCBR"

    if "_" in dataset_name:
        conf_model = conf_model[dataset_name.split("_")[0]]
    else:
        conf_model = conf_model[dataset_name]
    conf = {**conf_model, **conf_overall, **paras}
    conf["dataset"] = dataset_name
    conf["model"] = paras["model"]

    log_path = f"./logs/{conf['dataset']}/{conf['model']}"
    checkpoint_model_path = f"./checkpoints/{conf['dataset']}/{conf['model']}"
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    if not os.path.isdir(checkpoint_model_path):
        os.makedirs(checkpoint_model_path)

    setting = conf["model"] + "-" + conf["dataset"]
    if conf["info"] != "":
        setting = setting + "-" + conf["info"]
    log_path = log_path + "/" + setting + ".log"
    checkpoint_model_path = checkpoint_model_path + "/" + setting + ".pth"
    conf["log_path"] = log_path
    conf["checkpoint_model_path"] = checkpoint_model_path
    
    dataset = Datasets(conf)
    write_log(conf, log_path)

    conf["gpu"] = paras["gpu"]
    conf["info"] = paras["info"]
    conf["num_users"] = dataset.num_users
    conf["num_bundles"] = dataset.num_bundles
    conf["num_items"] = dataset.num_items
    os.environ['CUDA_VISIBLE_DEVICES'] = conf["gpu"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf["device"] = device
    
    if conf['model'] == 'DCBR':
        model = DCBR(conf, dataset.graphs).to(device)
    else:
        raise ValueError(f"Unimplemented model {conf['model']}")
    optimizer = torch.optim.Adam(model.parameters(), lr=conf["lr"], weight_decay=0)
    
    # Conditional Bundle Diffusion Model (CBDM)
    out_dims = conf["dims"] + [conf["num_bundles"]]
    in_dims = out_dims[::-1]
    denoise_model = DNN(in_dims, out_dims, conf["time_emb_dim"], norm=conf["norm"]).to(device)
    diffusion_model = GaussianDiffusion(conf["noise_scale"], conf["noise_min"], conf["noise_max"], conf["steps"]).to(device)
    cbdm_optimizer = torch.optim.Adam(denoise_model.parameters(), lr=conf["lr"], weight_decay=0)

    batch_cnt = len(dataset.train_loader)
    test_interval_bs = int(batch_cnt * conf["test_interval"])

    best_metrics = init_best_metrics(conf)
    best_epoch = 0
    best_content = None
    for epoch in range(conf['epochs']):
        ######### Denoising Uer-Bundle Graph ###########
        diffusionDataset = DiffusionDataset(torch.FloatTensor(dataset.graphs[0].A))
        diffusionLoader = torch.utils.data.DataLoader(diffusionDataset, batch_size=conf["batch_size_train"], shuffle=True, num_workers=0)
        total_steps = (diffusionDataset.__len__() + conf["batch_size_train"] - 1) // conf["batch_size_train"]
        pbar_diffusion = tqdm(enumerate(diffusionLoader), total=total_steps)
        for i, batch in pbar_diffusion:
            batch_user_bundle, batch_user_index = batch
            batch_user_bundle, batch_user_index = batch_user_bundle.to(device), batch_user_index.to(device)
            uEmbeds = model.getUserEmbeds().detach()
            bEmbeds = model.getBundleEmbeds().detach()
            
            cbdm_optimizer.zero_grad()
            elbo_loss, blcc_loss = diffusion_model.training_CBDM_losses(denoise_model, batch_user_bundle, uEmbeds, bEmbeds, batch_user_index)
            blcc_loss *= conf["lambda_0"]
            loss = elbo_loss + blcc_loss
            loss.backward()
            cbdm_optimizer.step()
            
            loss_scalar = loss.detach()
            elbo_loss_scalar = elbo_loss.detach()
            blcc_loss_scalar = blcc_loss.detach()
            pbar_diffusion.set_description(f'Diffusion Step: {i+1}/{total_steps} | loss: {loss_scalar:8.4f} | elbo_loss: {elbo_loss_scalar:8.4f} | blcc_loss: {blcc_loss_scalar:8.4f}')

        with torch.no_grad():
            u_list_ub = []
            b_list_ub = []
            edge_list_ub = []
            for _, batch in enumerate(diffusionLoader):
                batch_user_bundle, batch_user_index = batch
                batch_user_bundle, batch_user_index = batch_user_bundle.to(device), batch_user_index.to(device)
                denoised_batch = diffusion_model.p_sample(denoise_model, batch_user_bundle, conf["sampling_steps"], conf["sampling_noise"])
                _, indices_ = torch.topk(denoised_batch, k=conf["rebuild_k"])
                for i in range(batch_user_index.shape[0]):
                    for j in range(indices_[i].shape[0]): 
                        u_list_ub.append(int(batch_user_index[i].cpu().numpy()))
                        b_list_ub.append(int(indices_[i][j].cpu().numpy()))
                        edge_list_ub.append(1.0)
            u_list_ub = np.array(u_list_ub)
            b_list_ub = np.array(b_list_ub)
            edge_list_ub = np.array(edge_list_ub)
            denoised_ub_mat = sp.coo_matrix((edge_list_ub, (u_list_ub, b_list_ub)), shape=(conf["num_users"], conf["num_bundles"]), dtype=np.float32)
            adjacency_matrix = sp.bmat([[sp.csr_matrix((conf["num_users"], conf["num_users"])), denoised_ub_mat], [denoised_ub_mat.T, sp.csr_matrix((conf["num_bundles"], conf["num_bundles"]))]])
            adjacency_matrix = adjacency_matrix + sp.eye(adjacency_matrix.shape[0])
            row_sum = np.array(adjacency_matrix.sum(axis=1))
            d_inv = np.power(row_sum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            degree_matrix = sp.diags(d_inv)
            norm_adjacency = degree_matrix.dot(adjacency_matrix).dot(degree_matrix).tocoo()
            values = norm_adjacency.data
            indices = np.vstack((norm_adjacency.row, norm_adjacency.col))
            UB_propagation_graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(norm_adjacency.shape)).to(device)
            
        ################################################

        epoch_anchor = epoch * batch_cnt
        pbar = tqdm(enumerate(dataset.train_loader), total=len(dataset.train_loader))
        for batch_i, batch in pbar:
            model.train(True)
            optimizer.zero_grad()
            batch = [x.to(device) for x in batch]
            batch_anchor = epoch_anchor + batch_i

            bpr_loss, cl_loss = model(UB_propagation_graph, batch)
            loss = bpr_loss + cl_loss
            loss.backward()
            optimizer.step()

            loss_scalar = loss.detach()
            bpr_loss_scalar = bpr_loss.detach()
            cl_loss_scalar = cl_loss.detach()
            pbar.set_description(f'epoch: {epoch:3d} | loss: {loss_scalar:8.4f} | bpr_loss: {bpr_loss_scalar:8.4f} | cl_loss: {cl_loss_scalar:8.4f}')

            if (batch_anchor + 1) % test_interval_bs == 0:
                metrics = {}
                metrics["val"] = test(model, dataset.val_loader, conf)
                metrics["test"] = test(model, dataset.test_loader, conf)
                best_metrics, best_epoch, best_content = log_metrics(conf, model, metrics, log_path, checkpoint_model_path, epoch, best_metrics, best_epoch, best_content)
    write_log("="*26 + " BEST " + "="*26, log_path)
    write_log(best_content, log_path)


def init_best_metrics(conf):
    best_metrics = {}
    best_metrics["val"] = {}
    best_metrics["test"] = {}
    for key in best_metrics:
        best_metrics[key]["recall"] = {}
        best_metrics[key]["ndcg"] = {}
    for topk in conf['topk']:
        for key, res in best_metrics.items():
            for metric in res:
                best_metrics[key][metric][topk] = 0

    return best_metrics


def form_content(epoch, val_results, test_results, ks):
    content = f'     Epoch|'
    for k in ks:
        content += f' Recall@{k} |  NDCG@{k}  |'
    content += '\n'
    val_content = f'{epoch:10d}|'
    val_results_recall = val_results['recall']
    val_results_ndcg = val_results['ndcg']
    for k in ks:
        val_content += f'   {val_results_recall[k]:.4f}  |'
        val_content += f'   {val_results_ndcg[k]:.4f}  |'
    content += val_content + '\n'
    test_content = f'{epoch:10d}|'
    test_results_recall = test_results['recall']
    test_results_ndcg = test_results['ndcg']
    for k in ks:
        test_content += f'   {test_results_recall[k]:.4f}  |'
        test_content += f'   {test_results_ndcg[k]:.4f}  |'
    content += test_content
    return content


def log_metrics(conf, model, metrics, log_path, checkpoint_model_path, epoch, best_metrics, best_epoch, best_content):
    content = form_content(epoch, metrics["val"], metrics["test"], conf["topk"])
    write_log(content, log_path)

    topk_ = 20
    crt = f"top{topk_} as the final evaluation standard"
    write_log(crt, log_path)
    if metrics["val"]["recall"][topk_] > best_metrics["val"]["recall"][topk_] and metrics["val"]["ndcg"][topk_] > best_metrics["val"]["ndcg"][topk_]:
        state_dict = {
            "conf": conf,
            "cur_epoch": epoch,
            "content": content,
            "state_dict": model.state_dict(),
            "other_params": model.other_params()
        }
        torch.save(state_dict, checkpoint_model_path, pickle_protocol=4)
        saved = f"save the best checkpoint at the end of epoch {epoch}"
        write_log(saved, log_path)
        best_epoch = epoch
        best_content = content
        for topk in conf['topk']:
            for key, res in best_metrics.items():
                for metric in res:
                    best_metrics[key][metric][topk] = metrics[key][metric][topk]

    return best_metrics, best_epoch, best_content


def test(model, dataloader, conf):
    tmp_metrics = {}
    for m in ["recall", "ndcg"]:
        tmp_metrics[m] = {}
        for topk in conf["topk"]:
            tmp_metrics[m][topk] = [0, 0]

    device = conf["device"]
    model.eval()
    rs = model.propagate(test=True)
    for users, ground_truth_u_b, train_mask_u_b in dataloader:
        pred_b = model.evaluate(rs, users.to(device))
        pred_b -= 1e8 * train_mask_u_b.to(device)
        tmp_metrics = get_metrics(tmp_metrics, ground_truth_u_b, pred_b, conf["topk"])

    metrics = {}
    for m, topk_res in tmp_metrics.items():
        metrics[m] = {}
        for topk, res in topk_res.items():
            metrics[m][topk] = res[0] / res[1]

    return metrics


def get_metrics(metrics, grd, pred, topks):
    tmp = {"recall": {}, "ndcg": {}}
    for topk in topks:
        _, col_indice = torch.topk(pred, topk)
        row_indice = torch.zeros_like(col_indice) + torch.arange(pred.shape[0], device=pred.device, dtype=torch.long).view(-1, 1)
        is_hit = grd[row_indice.view(-1), col_indice.view(-1)].view(-1, topk)

        tmp["recall"][topk] = get_recall(pred, grd, is_hit, topk)
        tmp["ndcg"][topk] = get_ndcg(pred, grd, is_hit, topk)

    for m, topk_res in tmp.items():
        for topk, res in topk_res.items():
            for i, x in enumerate(res):
                metrics[m][topk][i] += x

    return metrics


def get_recall(pred, grd, is_hit, topk):
    epsilon = 1e-8
    hit_cnt = is_hit.sum(dim=1)
    num_pos = grd.sum(dim=1)

    # remove those test cases who don't have any positive items
    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = (hit_cnt / (num_pos + epsilon)).sum().item()

    return [nomina, denorm]


def get_ndcg(pred, grd, is_hit, topk):
    def DCG(hit, topk, device):
        hit = hit / torch.log2(torch.arange(2, topk + 2, device=device, dtype=torch.float))
        return hit.sum(-1)

    def IDCG(num_pos, topk, device):
        hit = torch.zeros(topk, dtype=torch.float)
        hit[:num_pos] = 1
        return DCG(hit, topk, device)

    device = grd.device
    IDCGs = torch.empty(1 + topk, dtype=torch.float)
    IDCGs[0] = 1  # avoid 0/0
    for i in range(1, topk + 1):
        IDCGs[i] = IDCG(i, topk, device)

    num_pos = grd.sum(dim=1).clamp(0, topk).to(torch.long)
    dcg = DCG(is_hit, topk, device)

    idcg = IDCGs[num_pos]
    ndcg = dcg / idcg.to(device)

    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = ndcg.sum().item()

    return [nomina, denorm]


if __name__ == "__main__":
    main()