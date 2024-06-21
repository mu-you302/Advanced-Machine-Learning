import torch
import argparse
from tqdm import tqdm
import numpy as np
import torch.backends.cudnn as cudnn
from config import cfg
from base import Tester


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, dest="gpu_ids", default="0,1")
    parser.add_argument("--test_epoch", type=str, dest="test_epoch", default=6)
    parser.add_argument(
        "--test_batch_size", type=int, dest="test_batch_size", default=32
    )
    parser.add_argument("--pretrained_model", type=str, dest="pretrained_model")
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if "-" in args.gpu_ids:
        gpus = args.gpu_ids.split("-")
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ",".join(map(lambda x: str(x), list(range(*gpus))))

    assert args.test_epoch, "Test epoch is required."
    return args


def main():

    args = parse_args()
    cfg.set_args(args.gpu_ids)
    cudnn.benchmark = True

    tester = Tester(args.test_epoch, args.pretrained_model)
    tester._make_batch_generator()
    tester._make_model()

    # eval_result = {}
    eval_result = [{"mpjpe": [], "pa_mpjpe": []} for i in range(17)]
    cur_sample_idx = 0
    for itr, (inputs, targets, meta_info) in enumerate(tqdm(tester.batch_generator)):

        # forward
        with torch.no_grad():
            inputs = {k: v.to(torch.device("cuda:0")) for k, v in inputs.items()}
            out = tester.model(inputs, targets, meta_info, "test")

        # save output
        out = {k: v.cpu().numpy() for k, v in out.items()}
        for k, v in out.items():
            batch_size = out[k].shape[0]
        out = [{k: v[bid] for k, v in out.items()} for bid in range(batch_size)]

        # evaluate
        cur_eval_result = tester._evaluate(out, cur_sample_idx)
        if cfg.testset == "Human36M":
            for i in range(17):
                eval_result[i]["mpjpe"] += cur_eval_result[i]["mpjpe"]
                eval_result[i]["pa_mpjpe"] += cur_eval_result[i]["pa_mpjpe"]
        else:
            for k, v in cur_eval_result.items():
                if k in eval_result:
                    eval_result[k] += v
                else:
                    eval_result[k] = v

        cur_sample_idx += len(out)

    tester._print_eval_result(eval_result)


if __name__ == "__main__":
    main()
