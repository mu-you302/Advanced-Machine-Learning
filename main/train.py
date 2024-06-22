import argparse
from config import cfg
import torch
from base import Trainer
import torch.backends.cudnn as cudnn
from tqdm import tqdm


def parse_args():
    """
        参数解析

    Args:
        --gpu: GPU编号
        --lr: 学习率
        --continue: 是否继续训练
        --end_epoch: 结束的epoch
        --train_batch_size: 训练批大小
        --parts: 要训练的身体部分
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, dest="gpu_ids", default="0,1")
    parser.add_argument("--lr", type=str, dest="lr", default=1e-5)
    parser.add_argument(
        "--continue", dest="continue_train", default=False, action="store_true"
    )
    parser.add_argument("--end_epoch", type=int, dest="end_epoch", default=7)
    parser.add_argument(
        "--train_batch_size", type=int, dest="train_batch_size", default=24
    )
    parser.add_argument("--parts", default="whole")
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if not args.lr:
        assert 0, "Please set learning rate"

    if "-" in args.gpu_ids:
        gpus = args.gpu_ids.split("-")
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ",".join(map(lambda x: str(x), list(range(*gpus))))

    return args


def main():

    # 解析参数并创建日志
    args = parse_args()
    cfg.set_args(args.gpu_ids, args.lr, args.continue_train)
    parts = args.parts
    cudnn.benchmark = True

    trainer = Trainer(parts=parts)
    trainer._make_batch_generator()
    trainer._make_model()

    # train
    for epoch in range(trainer.start_epoch, cfg.end_epoch):

        trainer.set_lr(epoch)
        trainer.tot_timer.tic()
        trainer.read_timer.tic()
        for itr, (inputs, targets, meta_info) in enumerate(trainer.batch_generator):
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()

            # forward
            trainer.optimizer.zero_grad()
            loss = trainer.model(inputs, targets, meta_info, "train")
            loss = {k: loss[k].mean() for k in loss}

            # backward
            sum(loss[k] for k in loss).backward()
            trainer.optimizer.step()
            trainer.gpu_timer.toc()
            screen = [
                "Epoch %d/%d itr %d/%d:"
                % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
                "lr: %g" % (trainer.get_lr()),
                "speed: %.2f(%.2fs r%.2f)s/itr"
                % (
                    trainer.tot_timer.average_time,
                    trainer.gpu_timer.average_time,
                    trainer.read_timer.average_time,
                ),
                "%.2fh/epoch"
                % (trainer.tot_timer.average_time / 3600.0 * trainer.itr_per_epoch),
            ]
            screen += ["%s: %.4f" % ("loss_" + k, v.detach()) for k, v in loss.items()]
            trainer.logger.info(" ".join(screen))

            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()

        trainer.save_model(
            {
                "epoch": epoch,
                "network": trainer.model.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
            },
            epoch,
        )


if __name__ == "__main__":
    main()
