"""
Author: wyh
Date: Nov 2024
"""

import argparse
import os
import sys
import shutil
import datetime
import logging
import importlib
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data_utils.S3DISDataLoader import S3DISDataset
import provider

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase','board', 'clutter']

class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {i: cat for i, cat in enumerate(seg_classes.keys())}




def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def parse_args():
    parser = argparse.ArgumentParser('TMF Training')
    parser.add_argument('--model', type=str, default='tmf', help='model name [default: tmf]')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size during training')
    parser.add_argument('--epoch', default=100, type=int, help='number of epochs')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root name')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=2048, help='point number')
    parser.add_argument('--step_size', type=int, default=50, help='lr decay step')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='lr decay gamma')
    parser.add_argument('--test_area', type=int, default=5, help='which area to use for test')
    parser.add_argument('--num_classes', type=int, default=13, help='number of semantic classes')
    parser.add_argument('--num_workers', type=int, default=4, help='number of dataloader workers')
    parser.add_argument('--block_size', type=float, default=5.0, help='block size for dataset loader')
    parser.add_argument('--input_channels', type=int, default=9, help='input feature channels for TMF')
    parser.add_argument('--resume', action='store_true', help='resume from best_model.pth if exists')
    return parser.parse_args()


def main(args):
    def log_string(out_str):
        logger.info(out_str)
        print(out_str)


    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)

    experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)

    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)

    checkpoints_dir = experiment_dir.joinpath('checkpoints')
    checkpoints_dir.mkdir(exist_ok=True)

    log_dir = experiment_dir.joinpath('logs')
    log_dir.mkdir(exist_ok=True)


    logger = logging.getLogger("TMF")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir / f'{args.model}.txt'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    log_string('PARAMETER ...')
    log_string(str(args))


    root = 'data/stanford_indoor3d/'
    NUM_CLASSES = args.num_classes
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size

    print("start loading training data ...")
    TRAIN_DATASET = S3DISDataset(
        split='train',
        data_root=root,
        num_point=NUM_POINT,
        test_area=args.test_area,
        block_size=args.block_size,
        sample_rate=1.0,
        transform=None
    )

    print("start loading test data ...")
    TEST_DATASET = S3DISDataset(
        split='test',
        data_root=root,
        num_point=NUM_POINT,
        test_area=args.test_area,
        block_size=args.block_size,
        sample_rate=1.0,
        transform=None
    )

    trainDataLoader = torch.utils.data.DataLoader(
        TRAIN_DATASET,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=lambda x: np.random.seed(x + int(time.time()))
    )

    testDataLoader = torch.utils.data.DataLoader(
        TEST_DATASET,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    weights = torch.Tensor(TRAIN_DATASET.labelweights).float().to(device)

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))


    MODEL = importlib.import_module(args.model)

    # save current scripts
    shutil.copy(f'models/{args.model}.py', str(experiment_dir))
    if os.path.exists('models/pointnet2_utils.py'):
        shutil.copy('models/pointnet2_utils.py', str(experiment_dir))

    classifier = MODEL.get_model(
        num_classes=NUM_CLASSES,
        input_channels=args.input_channels
    ).to(device)

    criterion = MODEL.get_loss().to(device)
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1 or classname.find('Conv1d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                torch.nn.init.xavier_normal_(m.weight.data)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                torch.nn.init.xavier_normal_(m.weight.data)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)

    start_epoch = 0
    best_iou = 0.0
    best_model_path = checkpoints_dir / 'best_model.pth'

    if args.resume and best_model_path.exists():
        checkpoint = torch.load(str(best_model_path), map_location=device)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            pass
        start_epoch = checkpoint.get('epoch', 0)
        best_iou = checkpoint.get('class_avg_iou', 0.0)
        log_string('Use pretrain model')
    else:
        log_string('No existing model, starting training from scratch...')
        classifier = classifier.apply(weights_init)


    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(
            classifier.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=args.decay_rate
        )

    # if resume, also try to load optimizer
    if args.resume and best_model_path.exists():
        checkpoint = torch.load(str(best_model_path), map_location=device)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECAY = 0.5
    MOMENTUM_DECAY_STEP = args.step_size

    global_epoch = start_epoch


    for epoch in range(start_epoch, args.epoch):
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))

        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECAY ** (epoch // MOMENTUM_DECAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        log_string('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))

        # ========================= TRAIN =========================
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0.0

        classifier.train()

        for i, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
            points = torch.Tensor(points)

            points = points.float().to(device)   # [B, N, C]
            target = target.long().to(device)    # [B, N]
            points = points.transpose(2, 1)      # [B, C, N]

            seg_pred, trans_feat = classifier(points)   # seg_pred: [B, N, NUM_CLASSES]

            loss = criterion(seg_pred, target, trans_feat, weights)

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                log_string("Loss has NaN or Inf values! Skip this batch.")
                continue

            loss.backward()
            optimizer.step()

            pred_choice = seg_pred.contiguous().view(-1, NUM_CLASSES).detach().cpu().data.max(1)[1].numpy()
            batch_label = target.view(-1).detach().cpu().numpy()

            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += batch_label.shape[0]
            loss_sum += loss.item()

        log_string('Training mean loss: %f' % (loss_sum / max(num_batches, 1)))
        log_string('Training accuracy: %f' % (total_correct / float(max(total_seen, 1))))


        if epoch % 5 == 0:
            logger.info('Save model...')
            savepath = str(checkpoints_dir / 'model.pth')
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch + 1,
                'class_avg_iou': best_iou,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        with torch.no_grad():
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0.0
            labelweights = np.zeros(NUM_CLASSES)
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

            classifier.eval()
            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))

            for i, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                points = torch.Tensor(points.data.numpy()).float().to(device)
                target = target.long().to(device)
                points = points.transpose(2, 1)

                seg_pred, trans_feat = classifier(points)   # [B, N, NUM_CLASSES]

                loss = criterion(seg_pred, target, trans_feat, weights)
                loss_sum += loss.item()

                pred_val = seg_pred.detach().cpu().numpy()          # [B, N, NUM_CLASSES]
                pred_val = np.argmax(pred_val, axis=2)              # [B, N]
                batch_label = target.detach().cpu().numpy()         # [B, N]

                correct = np.sum(pred_val == batch_label)
                total_correct += correct
                total_seen += batch_label.size

                tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                labelweights += tmp

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum(batch_label == l)
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum((pred_val == l) | (batch_label == l))

            labelweights = labelweights.astype(np.float32)
            if np.sum(labelweights) > 0:
                labelweights = labelweights / np.sum(labelweights)

            iou_list = []
            for l in range(NUM_CLASSES):
                iou = total_correct_class[l] / float(total_iou_deno_class[l] + 1e-6)
                iou_list.append(iou)
            mIoU = np.mean(iou_list)

            log_string('eval mean loss: %f' % (loss_sum / float(max(num_batches, 1))))
            log_string('eval point avg class IoU: %f' % mIoU)
            log_string('eval point accuracy: %f' % (total_correct / float(max(total_seen, 1))))
            log_string('eval point avg class acc: %f' % (
                np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=float) + 1e-6))
            ))

            iou_per_class_str = '------- IoU --------\n'
            for l in range(NUM_CLASSES):
                class_name = seg_label_to_cat[l] if l in seg_label_to_cat else str(l)
                iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                    class_name + ' ' * max(1, 16 - len(class_name)),
                    labelweights[l] if l < len(labelweights) else 0.0,
                    total_correct_class[l] / float(total_iou_deno_class[l] + 1e-6)
                )

            log_string(iou_per_class_str)
            log_string('Eval mean loss: %f' % (loss_sum / float(max(num_batches, 1))))
            log_string('Eval accuracy: %f' % (total_correct / float(max(total_seen, 1))))

            if mIoU >= best_iou:
                best_iou = mIoU
                logger.info('Save model...')
                savepath = str(checkpoints_dir / 'best_model.pth')
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch + 1,
                    'class_avg_iou': best_iou,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')

            log_string('Best mIoU: %f' % best_iou)

        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)