import os
import yaml
import torch
import argparse
from utility import Datasets
from models.DCBR import DCBR
from train import get_metrics


def get_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", type=str, default="0")
    parser.add_argument("-d", "--dataset", type=str, default="iFashion")
    parser.add_argument("-m", "--model", type=str, default="DCBR")
    parser.add_argument("-c", "--checkpoint", type=str, default="DCBR-iFashion.pth")
    args = parser.parse_args()
    return args


def quick_test(model, pretrained, test_loader, conf):
    tmp_metrics = {}
    for m in ["recall", "ndcg"]:
        tmp_metrics[m] = {}
        for topk in conf["topk"]:
            tmp_metrics[m][topk] = [0, 0]
    model.eval()
    rs = pretrained["restore_user_e"], pretrained["restore_bundle_e"]
    for users, ground_truth_u_b, train_mask_u_b in test_loader:
        pred_b = model.evaluate(rs, users.to(conf["device"]))
        pred_b -= 1e8 * train_mask_u_b.to(conf["device"])
        tmp_metrics = get_metrics(tmp_metrics, ground_truth_u_b, pred_b, conf["topk"])

    metrics = {}
    for m, topk_res in tmp_metrics.items():
        metrics[m] = {}
        for topk, res in topk_res.items():
            metrics[m][topk] = res[0] / res[1]

    return metrics


def test(model, test_loader, conf):
    tmp_metrics = {}
    for m in ["recall", "ndcg"]:
        tmp_metrics[m] = {}
        for topk in conf["topk"]:
            tmp_metrics[m][topk] = [0, 0]
    model.eval()
    users_rep, bundles_rep = model.propagate(test=True)
    rs = users_rep, bundles_rep
    for users, ground_truth_u_b, train_mask_u_b in test_loader:
        pred_b = model.evaluate(rs, users.to(conf["device"]))
        pred_b -= 1e8 * train_mask_u_b.to(conf["device"])
        tmp_metrics = get_metrics(tmp_metrics, ground_truth_u_b, pred_b, conf["topk"])

    metrics = {}
    for m, topk_res in tmp_metrics.items():
        metrics[m] = {}
        for topk, res in topk_res.items():
            metrics[m][topk] = res[0] / res[1]

    return metrics
    

def main():
    paras = get_cmd().__dict__
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
    conf["log_path"] = None
    dataset = Datasets(conf)
    conf["gpu"] = paras["gpu"]
    conf["num_users"] = dataset.num_users
    conf["num_bundles"] = dataset.num_bundles
    conf["num_items"] = dataset.num_items

    os.environ["CUDA_VISIBLE_DEVICES"] = conf["gpu"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf["device"] = device
    
    pretrained_model = torch.load(conf["checkpoint"])
    model = DCBR(conf, dataset.graphs).to(device)
    model.load_state_dict(pretrained_model["state_dict"])
    pretrained = pretrained_model["other_params"]

    metrics = quick_test(model, pretrained, dataset.test_loader, conf)
    # metrics = test(model, dataset.test_loader, conf)
    
    print(metrics)


if __name__ == '__main__':
    main()