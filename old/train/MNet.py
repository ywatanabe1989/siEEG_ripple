#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-11-01 18:44:22 (ywatanabe)"

import inspect
import matplotlib
import numpy as np
import torch
# from catboost import CatBoostClassifier, Pool

matplotlib.use("Agg")
import os
import sys

import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score

sys.path.append(".")
sys.path.append("./eeg_human_ripple_clf/externals/")
import mngs
from eeg_human_ripple_clf.models.MNet.MNet_100 import MNet_100
from eeg_human_ripple_clf import utils
import ranger
from tqdm import tqdm
# from eeg_human_ripple_clf.utils._DataLoaderFiller import DataLoaderFiller
# from eeg_human_ripple_clf.utils._MultiTaskLoss import MultiTaskLoss

# Fixes random seed
mngs.general.fix_seeds(seed=42, np=np, torch=torch)

# Configures matplotlib
mngs.plt.configure_mpl(plt)

def determine_save_dir(
    class_labels,
    model_name,
    window_size_sec,
    max_epochs,
):
    sdir = mngs.general.mk_spath("")
    comparison = mngs.general.connect_strs(class_labels, filler="_vs_")
    # comparison = "Ripple_vs_No-Ripple"
    sdir = (
        sdir + f"{os.getenv('USER')}/{comparison}/"
        f"_{model_name}_WindowSize-{window_size_sec}-sec_MaxEpochs_{max_epochs}"
        f"_{mngs.general.gen_timestamp()}/seg-level/"
    )
    return sdir


def load_model_and_model_config():
    from eeg_human_ripple_clf.models.MNet.MNet_100 import MNet_100 as Model
    # from eeg_human_ripple_clf.models.MNet.MNet_1000 import MNet_1000 as Model

    FPATH_MODEL = inspect.getfile(Model)
    FPATH_MODEL_CONF = FPATH_MODEL.replace(".py", ".yaml")
    MODEL_CONF = mngs.general.load(FPATH_MODEL_CONF)
    return Model, MODEL_CONF, FPATH_MODEL, FPATH_MODEL_CONF


def define_parameters(is_debug_mode=False):

    def _in_load_global_config(FPATH_GLOBAL_CONF):
        GLOBAL_CONF = {"SAMP_RATE": mngs.general.load(FPATH_GLOBAL_CONF)["SAMP_RATE_EEG"]}
        return GLOBAL_CONF

    def _in_load_model_and_model_config():
        return load_model_and_model_config()

    def _in_load_dataloader_config(FPATH_DL_CONF):
        DL_CONF = mngs.general.load(FPATH_DL_CONF)
        return DL_CONF
    
    ## Default config files

    # global
    FPATH_GLOBAL_CONF = "./eeg_human_ripple_clf/config/global.yaml"
    GLOBAL_CONF = _in_load_global_config(FPATH_GLOBAL_CONF)

    # model
    Model, MODEL_CONF, FPATH_MODEL, FPATH_MODEL_CONF = _in_load_model_and_model_config()

    ## dataloader
    FPATH_DL_CONF = "./eeg_human_ripple_clf/config/dataloader.yaml"
    DL_CONF = mngs.io.load(FPATH_DL_CONF)
    # # load_params
    # FPATH_LOAD_CONF = "eeg_human_ripple_clf/config/load_params.yaml"
    # LOAD_CONF = mngs.io.load(FPATH_LOAD_CONF)

    # # filler_params
    # FPATH_FILLER_CONF = "./eeg_human_ripple_clf/config/filler_params.yaml"
    # FILLER_CONF = mngs.io.load(FPATH_FILLER_CONF)

    default_confs = {
        "default_global_conf.yaml": GLOBAL_CONF,
        "default_model_conf.yaml": MODEL_CONF,
        "default_dl_conf.yaml": DL_CONF,
        # "default_load_conf.yaml": LOAD_CONF,
        # "default_filler_conf.yaml": FILLER_CONF,
    }

    ## Merges all default configs
    merged_conf = mngs.general.merge_dicts_wo_overlaps(
        GLOBAL_CONF, MODEL_CONF, DL_CONF,
    ) # LOAD_CONF, FILLER_CONF

    ## Verifies n_gpus
    # merged_conf["n_gpus"] = utils.verify_n_gpus(merged_conf["n_gpus"])
    merged_conf["n_gpus"] = mngs.ml.utils.verify_n_gpus(merged_conf["n_gpus"])    

    ## Updates merged_conf
    window_size_sec = float(
        np.round(merged_conf["window_size_pts"] / merged_conf["SAMP_RATE"], 1)
    )
    batch_size = (
        merged_conf["batch_size"] * merged_conf["n_gpus"]
        if not is_debug_mode
        else 16 * merged_conf["n_gpus"]
    )
    lr = float(merged_conf["lr"]) * merged_conf["n_gpus"]

    sdir = determine_save_dir(
        args.class_labels,
        Model.__name__,
        window_size_sec,
        args.max_epochs,
    )

    # ss, ee = re.search("2022-[0-9]{4}-[0-9]{4}", sdir).span()
    # if len(glob(sdir.replace(sdir[ss:ee], "*"))) > 0:
    #     print("\nSkipped\n")
    #     exit()

    sdir_wo_time = sdir

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    MONTAGE_8 = mngs.general.load("./config/global.yaml")["EEG_8_COMMON_CHANNELS"]
    MONTAGE_6 = MONTAGE_8[:6]
    merged_conf["montage"] = MONTAGE_6
    
    # merged_conf["montage"] = [
    #     f"{bi[0]}-{bi[1]}" for bi in merged_conf["montage"]
    # ]  # fixme
    # assert len(
    #     mngs.general.search(args.montages_to_mask, merged_conf["montage"])[0]
    # ) == len(args.montages_to_mask)
    merged_conf.update(
        {
            "MAX_EPOCHS": args.max_epochs,
            "device_type": device.type,
            "device_index": device.index,
            "sdir": sdir,
            "is_debug_mode": is_debug_mode,
            "window_size_sec": window_size_sec,
            "batch_size": batch_size,
            "lr": lr,
            "class_labels": args.class_labels,
        }
    )

    ## Saves files to reproduce
    files_to_reproduce = [
        mngs.io.get_this_fpath(when_ipython="/dev/null"),
        FPATH_MODEL,
        FPATH_MODEL_CONF,
        FPATH_DL_CONF,
        # FPATH_LOAD_CONF,
        # FPATH_FILLER_CONF,
    ]

    for f in files_to_reproduce:
        mngs.io.save(f, merged_conf["sdir"])

    return merged_conf, default_confs, Model

def init_a_model(Model, config):
    model = Model(config)

    if config["n_gpus"] > 1:
        model = nn.DataParallel(model)
        print(f'Let\'s use {config["n_gpus"]} GPUs!')
    return model

def train_and_validate(
    dlf,
    i_fold,
    Model,
    merged_conf,
):
    print(f"\n {'-'*40} fold#{i_fold} starts. {'-'*40} \n")

    # model
    device = f"{merged_conf['device_type']}:{merged_conf['device_index']}"
    dlf.fill(i_fold, reset_fill_counter=True)  # to get n_subjs_tra

    # class_weight
    counts = dlf.sample_counts
    neg_weight = counts[1] / counts.sum()
    pos_weight = counts[0] / counts.sum()
    class_weight = torch.tensor(np.array([neg_weight, pos_weight]).astype(np.float32))


    merged_conf["n_subjs_tra"] = len(
        dlf.subs_tra
    )  # len(np.unique(dlf.dl_tra.dataset.arrs_list[-1]))
    model = init_a_model(Model, merged_conf).to(device)

    mtl = utils.MultiTaskLoss(are_regression=[False, False]).to(device)
    optimizer = ranger.Ranger(
        list(model.parameters()) + list(mtl.parameters()), lr=merged_conf["lr"]
    )

    # starts the current fold's loop
    i_global = 0
    lc_logger = mngs.ml.LearningCurveLogger()
    early_stopping = utils.EarlyStopping(patience=50, verbose=True)
    for i_epoch, epoch in enumerate(tqdm(range(merged_conf["MAX_EPOCHS"]))):

        dlf.fill(i_fold, reset_fill_counter=False)

        step_str = "Validation"
        for i_batch, batch in enumerate(dlf.dl_val):
            _, loss_diag_val = utils.base_step(
                step_str,
                model,
                mtl,
                batch,
                device,
                i_fold,
                i_epoch,
                i_batch,
                i_global,
                lc_logger,
                print_batch_interval=False,
                class_weight=class_weight,
            )
        lc_logger.print(step_str)

        step_str = "Training"
        for i_batch, batch in enumerate(dlf.dl_tra):
            optimizer.zero_grad()
            loss, _ = utils.base_step(
                step_str,
                model,
                mtl,
                batch,
                device,
                i_fold,
                i_epoch,
                i_batch,
                i_global,
                lc_logger,
                print_batch_interval=False,
            )
            loss.backward()
            optimizer.step()
            i_global += 1
        lc_logger.print(step_str)

        bACC_val = np.array(lc_logger.logged_dict["Validation"]["bACC_diag_plot"])[
            np.array(lc_logger.logged_dict["Validation"]["i_epoch"]) == i_epoch
        ].mean()

        model_spath = (
            merged_conf["sdir"]
            + f"checkpoints/model_fold#{i_fold}_epoch#{i_epoch:03d}.pth"
        )
        mtl_spath = model_spath.replace("model_fold", "mtl_fold")
        spaths_and_models_dict = {model_spath: model, mtl_spath: mtl}

        # early_stopping(loss_diag_val, spaths_and_models_dict, i_epoch, i_global)
        early_stopping(-bACC_val, spaths_and_models_dict, i_epoch, i_global)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    true_label_val = np.hstack(
        lc_logger.get_x_of_i_epoch(
            "true_label_diag", "Validation", early_stopping.i_epoch
        )
    )

    pred_proba_val = np.vstack(
        lc_logger.get_x_of_i_epoch(
            "pred_proba_diag", "Validation", early_stopping.i_epoch
        )
    )
    pred_class_val = pred_proba_val.argmax(axis=-1)

    bACC_val_best = balanced_accuracy_score(true_label_val, pred_class_val)

    to_test = [i_global, early_stopping, dlf, lc_logger, merged_conf]
    return bACC_val_best, to_test
    # return None, None


def test(
    i_fold, Model, reporter, i_global, early_stopping, dlf, lc_logger, merged_conf
):
    model = init_a_model(Model, merged_conf)
    model.load_state_dict(
        torch.load(list(early_stopping.spaths_and_models_dict.keys())[0])
    )
    mtl = utils.MultiTaskLoss(are_regression=[False, False])
    mtl.load_state_dict(
        torch.load(list(early_stopping.spaths_and_models_dict.keys())[1])
    )

    device = f"{merged_conf['device_type']}:{merged_conf['device_index']}"
    model.to(device)
    mtl.to(device)

    step_str = "Test"
    for i_batch, batch in enumerate(dlf.dl_tes):

        _, _ = utils.base_step(
            step_str,
            model,
            mtl,
            batch,
            device,
            i_fold,
            early_stopping.i_epoch,
            i_batch,
            early_stopping.i_global,
            lc_logger,
            print_batch_interval=False,
        )
    lc_logger.print(step_str)

    ## Evaluate on Test dataset
    true_class_tes = np.hstack(lc_logger.dfs["Test"]["true_label_diag"])
    pred_proba_tes = np.vstack(lc_logger.dfs["Test"]["pred_proba_diag"])
    pred_class_tes = pred_proba_tes.argmax(axis=-1)

    if true_class_tes.ndim == 1:
        true_class_tes = true_class_tes[..., np.newaxis]
        pred_class_tes = pred_class_tes[..., np.newaxis]

    # labels = list(dlf.cv_dict["label_int_2_conc_class_dict"].values())  # dlf.disease_types
    labels = ["NoN-Ripple", "Ripple"]

    reporter.calc_metrics(
        true_class_tes, pred_class_tes, pred_proba_tes, labels=labels, i_fold=i_fold
    )

    ## learning curves
    plt_config_dict = dict(
        dpi=300,
        figsize=(16.2, 10),
        # figscale=1.0,
        figscale=2.0,
        fontsize=16,
        labelsize="same",
        legendfontsize="xx-small",
        tick_size="auto",
        tick_width="auto",
        hide_spines=False,
    )
    lc_fig = lc_logger.plot_learning_curves(
        plt,
        title=(
            f"fold#{i_fold}\nmax epochs: {merged_conf['MAX_EPOCHS']}\n"
            f"window_size: {merged_conf['window_size_sec']} [sec]"
        ),
        plt_config_dict=plt_config_dict,
    )
    reporter.add("learning_curve", lc_fig)

    return reporter


def main(is_debug_mode=False):
    merged_conf, default_confs, Model = define_parameters(is_debug_mode=is_debug_mode)

    sys.stdout, sys.stderr = mngs.general.tee(sys, sdir=merged_conf["sdir"])
    mngs.general.fix_seeds(np=np, torch=torch)

    reporter = mngs.ml.ClassificationReporter(merged_conf["sdir"])

    dlf = utils.DataLoaderFiller(**merged_conf)

    # dlf = DataLoaderFiller(
    #     "./data/BIDS_Osaka",
    #     args.disease_types,
    #     drop_cMCI=True,
    # )

    # k-fold CV loop
    for i_fold in range(merged_conf["n_repeat"]):

        bACC_val_best, to_test = train_and_validate(
            dlf,
            i_fold,
            Model,
            merged_conf,
        )
        reporter = test(i_fold, Model, reporter, *to_test)

    ## Saves the results
    reporter.summarize()
    reporter.save(meta_dict={**default_confs, "merged_conf.yaml": merged_conf})
    confmat_plt_config = dict(
        figsize=(15, 15),
        # labelsize=8,
        # fontsize=6,
        # legendfontsize=6,
        figscale=2,
        tick_size=0.8,
        tick_width=0.2,
    )

    sci_notation_kwargs = dict(
        order=3,  # 1
        fformat="%1.0d",
        scilimits=(-3, 3),
        x=False,
        y=True,
    )  # "%3.1f"

    reporter.plot_and_save_conf_mats(
        plt,
        extend_ratio=0.8,
        confmat_plt_config=confmat_plt_config,
        sci_notation_kwargs=sci_notation_kwargs,
    )


if __name__ == "__main__":
    import argparse

    import mngs

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-cl",
        "--class_labels",
        default=["NoN-Ripple", "Ripple"],
        nargs="*",
        help=" ",
    )
    # ap.add_argument("-ws", "--window_size_sec", default=2, type=int, help=" ")
    ap.add_argument("-me", "--max_epochs", default=20, type=int, help=" ")  # 50    
    args = ap.parse_args()

    ws_ms = 500
    tau_ms = 0
    # main(ws_ms, tau_ms)
    main()

