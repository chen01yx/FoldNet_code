import argparse
from garmentds.foldenv.fold_learn import _quick_find, is_success_demo, load_json
import tqdm
from collections import defaultdict
import omegaconf
import os
import pprint


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--error_threshold", "-e", default=1.0, type=float)
    parser.add_argument("--dir", "-d", type=str, nargs="+")
    parser.add_argument("--ignore_gt_ik_fail", "-i", action="store_true")
    args = parser.parse_args()
    return args


def split_to_tuple(x: str):
    x = x.split("/")
    return tuple([int(d) if d.isnumeric() else d for d in x])


def main(args: dict):
    find_result = _quick_find(args["dir"], "meta_info_demo.json", use_cache=True)
    find_result = sorted(find_result, key=split_to_tuple)

    fail_reason = defaultdict(int)
    all_garments = {}
    succ = 0
    fail = 0
    for meta_info_demo_json in find_result:
        is_succ, extra_info = is_success_demo(meta_info_demo_json, err_th=args["error_threshold"])
        for reason in extra_info["fail_reason"]:
            fail_reason[reason] += 1
        print(f"{meta_info_demo_json} is_succ:{is_succ} err:{extra_info['err']} fail_reason:{extra_info['fail_reason']}")
        env_cfg = omegaconf.OmegaConf.load(os.path.join(os.path.dirname(meta_info_demo_json), "..", ".hydra", "env_cfg.yaml"))
        cloth_obj_path = os.path.relpath(env_cfg.cloth_obj_path, ".")
        if cloth_obj_path not in all_garments:
            all_garments[cloth_obj_path] = {"succ": 0, "fail": 0}
        if ("gt_ik_fail" not in extra_info["fail_reason"]) or not (args["ignore_gt_ik_fail"]):
            if is_succ:
                succ += 1
                all_garments[cloth_obj_path]["succ"] += 1
            else:
                fail += 1
                all_garments[cloth_obj_path]["fail"] += 1
    
    print("==============================================================")
    for cloth_obj_path, cloth_info in sorted(list(all_garments.items()), key=lambda x: split_to_tuple(x[0])):
        print(f"{cloth_obj_path} success rate: {cloth_info['succ']} / {cloth_info['succ'] + cloth_info['fail']} = {cloth_info['succ'] / (cloth_info['succ'] + cloth_info['fail'])}")
    info_path = os.path.join(args["dir"][0], ".tmp/info.json")
    if os.path.exists(info_path):
        data = load_json(info_path)
        print("ckpt:", data.get("args", dict()).get("ckpt", None))
    print(f"total success rate @ {args['error_threshold']} : {succ} / {succ + fail} = {succ / (succ + fail)} ")
    print(fail_reason)


if __name__ == "__main__":
    args = get_args()
    main(vars(args))