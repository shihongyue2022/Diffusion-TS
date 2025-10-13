#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import itertools
from typing import Tuple, Optional

import numpy as np
import torch

from engine.logger import Logger
from engine.solver import Trainer
from Data.build_dataloader import build_dataloader, build_dataloader_cond
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one
from Utils.io_utils import load_yaml_config, seed_everything, merge_opts_to_config, instantiate_from_config


# ------------------ helpers ------------------ #
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _align_AT(arr: np.ndarray, A: int, T: int) -> Optional[np.ndarray]:
    """尽量将二维数组对齐为 (A,T)。"""
    if arr.ndim != 2:
        return None
    h, w = arr.shape
    # 完全匹配
    if (h, w) == (A, T):
        return arr
    if (h, w) == (T, A):
        return arr.T
    # 一维匹配 + 时间尾对齐
    if h == A and w > T:
        return arr[:, -T:]
    if w == A and h > T:
        return arr[-T:, :].T
    # 只能返回 None，交给上层兜底
    return None

def _guess_ast_from_3d(arr: np.ndarray, A: int, T: int) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    尝试将 3D 数组排列到 [A,S,T]，返回 (arr_ast, S)；失败则 (None, None)。
    允许 S/T 交换（只要能把 A 与 T 对上，就把剩下一维当 S）。
    """
    if arr.ndim != 3:
        return None, None
    for perm in itertools.permutations(range(3), 3):
        a, s, t = arr.shape[perm[0]], arr.shape[perm[1]], arr.shape[perm[2]]
        if a == A and t == T:
            return np.transpose(arr, perm), s
    # 放宽：只要 A 轴对上，T 用“最接近 T 的一维”去猜
    for perm in itertools.permutations(range(3), 3):
        a, x, y = arr.shape[perm[0]], arr.shape[perm[1]], arr.shape[perm[2]]
        if a == A:
            # 优先选与 T 最接近的一维当 T
            if abs(x - T) <= abs(y - T):
                s, t = y, x
                ast = np.transpose(arr, (perm[0], perm[2], perm[1]))
            else:
                s, t = x, y
                ast = np.transpose(arr, perm)
            # 若 t > T 取尾部
            if t > T:
                ast = ast[:, :, -T:]
                t = T
            if t == T:
                return ast, s
    return None, None

def _save_all(save_dir: str, name: str, mode: str, samples, var_num: int, window: int):
    """统一保存：raw / auto / ast + meta.json"""
    arr = _to_numpy(samples)
    _ensure_dir(save_dir)

    # 1) 原样保存
    raw_fp = os.path.join(save_dir, "pred_raw.npy")
    np.save(raw_fp, arr)

    picked = "raw"
    ast_fp = None
    auto_fp = os.path.join(save_dir, "pred_auto.npy")  # 兼容 eval 端默认
    meta = {"name": name, "mode": mode, "var_num": int(var_num), "window": int(window),
            "raw_shape": list(arr.shape), "files": {"raw": raw_fp}}

    # 2) 生成 auto（2D/3D 都可）
    auto_arr = arr
    if auto_arr.ndim == 3:
        # 尝试映射到 [A,S,T]，映射成功则也存一份 ast
        ast_arr, S = _guess_ast_from_3d(auto_arr, var_num, window)
        if ast_arr is not None:
            ast_fp = os.path.join(save_dir, "pred_ast.npy")
            np.save(ast_fp, ast_arr)  # [A,S,T]
            picked = "ast"
            meta["files"]["ast"] = ast_fp
            meta["ast_shape"] = list(ast_arr.shape)
            meta["S_inferred"] = int(S)
        # 不论是否成功，auto 先存 raw 的副本（eval 也能识别）
        np.save(auto_fp, auto_arr)
        meta["files"]["auto"] = auto_fp
        meta["auto_shape"] = list(auto_arr.shape)

    elif auto_arr.ndim == 2:
        at = _align_AT(auto_arr, var_num, window)
        if at is None:
            # 强行兜底：若最后一维等于 var_num，当成 (T,A)，否则当 (A,T)；时间维尾部裁剪到 window
            if auto_arr.shape[-1] == var_num:
                at = auto_arr
                if at.shape[0] > window:
                    at = at[-window:, :]
                at = at.T  # (A,T)
            else:
                at = auto_arr
                if at.shape[1] > window:
                    at = at[:, -window:]
        # 存 auto = 2D
        np.save(auto_fp, at)
        meta["files"]["auto"] = auto_fp
        meta["auto_shape"] = list(at.shape)
        # 同时提供 ast（扩一个 S=1）
        ast_arr = at[:, None, :]  # [A,1,T]
        ast_fp = os.path.join(save_dir, "pred_ast.npy")
        np.save(ast_fp, ast_arr)
        picked = "ast"
        meta["files"]["ast"] = ast_fp
        meta["ast_shape"] = list(ast_arr.shape)
        meta["S_inferred"] = 1

    else:
        # 其他维度先如实保存 auto，eval 会尽量识别
        np.save(auto_fp, auto_arr)
        meta["files"]["auto"] = auto_fp
        meta["auto_shape"] = list(auto_arr.shape)

    meta["picked"] = picked
    meta_fp = os.path.join(save_dir, "pred_meta.json")
    with open(meta_fp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    meta["files"]["meta"] = meta_fp

    print(f"[save] raw -> {raw_fp}  shape={arr.shape}")
    if "auto" in meta["files"]:
        print(f"[save] auto-> {meta['files']['auto']}  shape={meta.get('auto_shape')}")
    if "ast" in meta["files"]:
        print(f"[save] ast -> {meta['files']['ast']}  shape={meta.get('ast_shape')}")
    print(f"[save] meta-> {meta_fp}")


# ------------------ argparse ------------------ #
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training / Sampling for Diffusion-TS')
    parser.add_argument('--name', type=str, default=None)

    parser.add_argument('--config_file', type=str, default=None,
                        help='path of config file')
    parser.add_argument('--output', type=str, default=None,
                        help='base directory to save results; 若未指定，则使用 config.solver.results_folder')
    parser.add_argument('--tensorboard', action='store_true',
                        help='use tensorboard for logging')

    # random
    parser.add_argument('--cudnn_deterministic', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--gpu', type=int, default=None)

    # train / sample
    parser.add_argument('--train', action='store_true', default=False, help='Train or Test.')
    parser.add_argument('--sample', type=int, default=0, choices=[0, 1], help='Condition or Uncondition.')
    parser.add_argument('--mode', type=str, default='infill', help='infill | predict')
    parser.add_argument('--milestone', type=int, default=10)

    parser.add_argument('--missing_ratio', type=float, default=0., help='Ratio of Missing Values.')
    parser.add_argument('--pred_len', type=int, default=0, help='Length of Predictions.')

    # modify config on-the-fly
    parser.add_argument('opts', help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    # 先占位，真正的 save_dir 等读取完 YAML 再覆盖
    args.save_dir = None
    return args


def main():
    args = parse_args()

    if args.seed is not None:
        seed_everything(args.seed)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    # 读取配置并合并命令行覆盖
    config = load_yaml_config(args.config_file)
    config = merge_opts_to_config(config, args.opts)

    # 决定输出目录：优先 --output，否则用 solver.results_folder
    results_folder = (config.get('solver', {}) or {}).get('results_folder', './Checkpoints_stock')
    base_out = args.output if (args.output not in [None, '']) else results_folder
    # 与权重保持一致：<base>/<name> 结构
    args.save_dir = os.path.join(base_out, f'{args.name}')
    _ensure_dir(args.save_dir)

    # 日志器
    logger = Logger(args)
    logger.save_config(config)

    # 模型与数据
    model = instantiate_from_config(config['model']).cuda()
    if args.sample == 1 and args.mode in ['infill', 'predict']:
        test_dataloader_info = build_dataloader_cond(config, args)
    dataloader_info = build_dataloader(config, args)
    trainer = Trainer(config=config, args=args, model=model, dataloader=dataloader_info, logger=logger)

    # =============== 训练 =============== #
    if args.train:
        trainer.train()
        return

    # =============== 采样（有条件预测：infill / predict） =============== #
    if args.sample == 1 and args.mode in ['infill', 'predict']:
        trainer.load(args.milestone)
        dataloader, dataset = test_dataloader_info['dataloader'], test_dataloader_info['dataset']
        coef = config['dataloader']['test_dataset']['coefficient']
        stepsize = config['dataloader']['test_dataset']['step_size']
        sampling_steps = config['dataloader']['test_dataset']['sampling_steps']

        model.eval()
        with torch.no_grad():
            samples, *_ = trainer.restore(
                dataloader,
                [dataset.window, dataset.var_num],
                coef, stepsize, sampling_steps
            )
        # 反归一化（若启用 auto_norm）
        if getattr(dataset, "auto_norm", False):
            samples = unnormalize_to_zero_to_one(samples)

        # 统一保存：raw / auto / ast + meta
        _save_all(args.save_dir, args.name, args.mode, samples, dataset.var_num, dataset.window)
        return

    # =============== 采样（无条件） =============== #
    trainer.load(args.milestone)
    dataset = dataloader_info['dataset']
    model.eval()
    with torch.no_grad():
        samples = trainer.sample(num=len(dataset), size_every=2001, shape=[dataset.window, dataset.var_num])
    if getattr(dataset, "auto_norm", False):
        samples = unnormalize_to_zero_to_one(samples)

    _save_all(args.save_dir, args.name, "uncond", samples, dataset.var_num, dataset.window)


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        # 把异常打印清楚，便于你定位
        import traceback
        print("[FATAL] main.py uncaught exception:", e)
        traceback.print_exc()
        raise
