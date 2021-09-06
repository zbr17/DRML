import sys
import os
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "relational_dml"
    )
)
import logging
logging.getLogger().setLevel(logging.INFO)
import torch.backends.cudnn as cudnn
import logging
import yaml
import argparse
from copy import deepcopy
import numpy as np 
import torch
import time

from gedml.launcher.creators import ConfigHandler
from gedml.launcher.misc import utils

from source import models
from source import collectors

parser = argparse.ArgumentParser(description="Deep Relational Metric Learning")
# setting
parser.add_argument("--save_name", type=str, default="TEST")
parser.add_argument("--not_test", action="store_true", default=False)
parser.add_argument("--phase", type=str, default="evaluate")
parser.add_argument("--delete_old", action="store_true", default=False)
parser.add_argument("--device", type=int, nargs='+', default=[0])
parser.add_argument("--num_worders", type=int, default=8)
parser.add_argument("--setting", type=str, default="baseline")
parser.add_argument("--dataset", type=str, default="cub200")
parser.add_argument("--interval", type=int, default=1)
parser.add_argument("--is_resume", action="store_true", default=False)
parser.add_argument("--device_type", type=str, default="DP")
parser.add_argument("--warmup", type=int, default=1)
# model and loss
parser.add_argument("--backbone", type=str, default="bninception", help="bninception / resnet50")
parser.add_argument("--embedding_size", type=int, default=512)
parser.add_argument("--num_classes", type=int, default=100)
parser.add_argument("--branch_num", type=int, default=4) # TODO: DON'T MODIFY THIS PARAMETER
parser.add_argument("--weight_repre_loss", type=float, default=1.0)
parser.add_argument("--weight_recon_loss", type=float, default=1.0)
parser.add_argument("--reg_weight", type=float, default=0.0)
parser.add_argument("--centers_per_class", type=int, default=1)
parser.add_argument("--features_dim", type=int, default=1024)
parser.add_argument("--margin_alpha", type=float, default=0.8)
parser.add_argument("--margin_beta", type=float, default=0.4)
parser.add_argument("--proxy_margin", type=float, default=0.1)
parser.add_argument("--proxy_alpha", type=float, default=32)
parser.add_argument("--triplet_margin", type=float, default=1)
parser.add_argument("--cosface_margin", type=float, default=0.35)
parser.add_argument("--cosface_scale", type=float, default=64)
# selectors
parser.add_argument("--hardneg_cutoff", type=float, default=0.5)
# optimizer
parser.add_argument("--lr_trunk", type=float, default=1e-5)
parser.add_argument("--lr_embedder", type=float, default=0.0001)
parser.add_argument("--lr_loss", type=float, default=0.01)
parser.add_argument("--lr_proxy", type=float, default=0.01)

# schedulers
parser.add_argument("--gamma", type=float, default=0.5)
parser.add_argument("--step_size", type=int, default=1)
# training 
parser.add_argument("--batch_size", type=int, default=180)
# testing
parser.add_argument("--test_batch_size", type=int, default=180)
parser.add_argument("--splits_to_eval", type=str, nargs='+', default=["test"])
## recorder
parser.add_argument("--use_wandb", action="store_true", default=False)

opt = parser.parse_args()
opt.is_resume = True
opt.delete_old = False

div_constant = int(opt.features_dim / opt.embedding_size) # default: 2
opt.input_size = int(opt.features_dim / opt.branch_num) # default: 256
opt.hidden_embeddings_size = int(opt.input_size / div_constant) # default: 128

### Hyper-parameters
link_root = "./source/config/link"
link_path = os.path.abspath(os.path.join(link_root, "link_" + opt.setting + ".yaml"))
convert_root = "./source/config/convert"
convert_path = os.path.abspath(os.path.join(convert_root, "convert_" + opt.setting + ".yaml"))
opt.save_root = "../../experiments/relational_dml/2021-06-18/"
save_name = opt.save_name
save_path = os.path.abspath(os.path.join(opt.save_root, opt.save_name))
wrapper_path = os.path.abspath("./source/config/wrapper")
params_path = os.path.abspath("./source/config/param")
warmup = opt.warmup
dataset = opt.dataset
backbone = opt.backbone

phase = opt.phase
is_test = not opt.not_test
setting = opt.setting
start_epoch = 0
is_save = True
cudnn.deterministic = True
cudnn.benchmark = True

##########################
### Start cross-validation
##########################

from sklearn.model_selection import KFold
total_index = np.arange(100)
n_split = 4

save_path_list = [
    save_path + "_split_{}".format(idx)
    for idx in range(n_split)
]

output_metrics = []

for idx, (train_split, test_split) in enumerate(KFold(n_splits=n_split).split(total_index)):
    logging.info("Split: {}".format(idx))
    opt.save_path = save_path_list[idx]

    # get confighandler
    config_handler = ConfigHandler(
        link_path=link_path,
        params_path=params_path,
        wrapper_path=wrapper_path,
        convert_path=convert_path,
        is_confirm_first=False
    )

    # register packages
    config_handler.register_packages("models", models)
    config_handler.register_packages("collectors", collectors)

    # initiate params_dict
    config_handler.get_params_dict(
        modify_link_dict={
            "datasets": [
                {"train": "{}_train.yaml".format(dataset)},
                {"test": "{}_test.yaml".format(dataset)}
            ],
            "models": [
                {"trunk": "{}.yaml".format(backbone)},
            ]
        }
    )

    if idx == 0:
        # delete redundant options
        convert_opt_list = list(config_handler.convert_dict.keys())
        for k in list(opt.__dict__.keys()):
            if k not in convert_opt_list:
                delattr(opt, k)

    # modify parameters
    objects_dict = config_handler.create_all(opt.__dict__)

    # get manager and tester and evaluator
    manager = utils.get_default(objects_dict, "managers")
    tester = utils.get_default(objects_dict, "testers")
    evaluator = utils.get_default(objects_dict, "evaluators")
    
    # start
    manager.prepare()
    manager.maybe_resume()
    tester.initiate_datasets()
    tester.set_to_eval()
    outputs = {}
    with torch.no_grad():
        for k, v in tester.datasets.items():
            # get the dataset loader
            tester.initiate_dataloader(dataset=v)
            # get the embeddings
            tester.get_embeddings()
            metrics_dict = evaluator.get_accuracy(
                tester.embeddings,
                tester.embeddings,
                tester.labels,
                tester.labels,
                True,
                device_ids=[0],
            )
            output_metrics.append(metrics_dict)


save_f = open("/home/zbr/code/relational_dml/results/{}-mean.txt".format(save_name), mode="w")
output_list = []

for k in output_metrics[0].keys():
    value = [
        output_metrics[idx][k]
        for idx in range(n_split)
    ]
    value = sum(value) / n_split
    str_output = "{}: {}\n".format(k, value)
    print(str_output)
    output_list.append(str_output)
save_f.writelines(output_list)
save_f.close()



