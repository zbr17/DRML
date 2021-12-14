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

from src import models
from src import collectors

parser = argparse.ArgumentParser(description="Deep Relational Metric Learning")
# setting
parser.add_argument("--save_name", type=str, default="TEST")
parser.add_argument("--not_test", action="store_true", default=False)
parser.add_argument("--phase", type=str, default="train")
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
if opt.is_resume:
    opt.delete_old = False

div_constant = int(opt.features_dim / opt.embedding_size) # default: 2
opt.input_size = int(opt.features_dim / opt.branch_num) # default: 256
opt.hidden_embeddings_size = int(opt.input_size / div_constant) # default: 128

### Hyper-parameters
link_root = "./src/drml/config/link"
link_path = os.path.abspath(os.path.join(link_root, "link_" + opt.setting + ".yaml"))
convert_root = "./src/drml/config/convert"
convert_path = os.path.abspath(os.path.join(convert_root, "convert_" + opt.setting + ".yaml"))
opt.save_root = "../../experiments/relational_dml/{}/".format(time.strftime("%Y-%m-%d", time.localtime()))
save_path = os.path.abspath(os.path.join(opt.save_root, opt.save_name))
wrapper_path = os.path.abspath("./src/drml/config/wrapper")
params_path = os.path.abspath("./src/drml/config/param")
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
from gedml.core.samplers import MPerClassFullSampler

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

    # change datasets
    ori_dataset = objects_dict["datasets"]["train"]
    labels = ori_dataset.labels
    indices_dict = {idx: torch.where(torch.Tensor(labels)==idx)[0] for idx in range(100)}
    train_indices = torch.cat([indices_dict[idx] for idx in train_split])
    test_indices = torch.cat([indices_dict[idx] for idx in test_split])

    train_dataset = torch.utils.data.Subset(deepcopy(ori_dataset), train_indices)
    test_dataset = torch.utils.data.Subset(deepcopy(ori_dataset), test_indices)

    train_trans = objects_dict["transforms"]["train"]
    test_trans = objects_dict["transforms"]["test"]

    train_dataset.dataset.transform = train_trans
    test_dataset.dataset.transform = test_trans

    objects_dict["datasets"]["train"] = train_dataset
    objects_dict["datasets"]["test"] = test_dataset

    # temporary mathods
    if "nosampler" not in setting:
        if "triplet" in setting or "margin" in setting:
            sub_labels = []
            for i in range(len(train_dataset)):
                sub_labels.append(train_dataset.__getitem__(i)["labels"])
            
            sub_sampler = MPerClassFullSampler(
                labels=sub_labels,
                m=2,
                batch_size=32,
            )
            objects_dict["samplers"]["train"] = sub_sampler

    # get manager
    manager = utils.get_default(objects_dict, "managers")

    warm_up_list = (
        ["loss", "embedder"]
        if "baseline" in setting
        else ["embedder", "loss1", "loss2", "loss3", "loss4", "loss"]
    )
    # start
    manager.run(
        phase=phase,
        start_epoch=start_epoch,
        is_test=is_test,
        is_save=is_save,
        warm_up=warmup,
        warm_up_list=warm_up_list
    )
