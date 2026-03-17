
import argparse
import os
import sys
import importlib
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data_utils.S3DISDataLoader import ScannetDatasetWholeScene


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


classes = [
    'ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
    'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter'
]

S3DIS_COLOR_MAP = {
    0: [233, 229, 107],
    1: [95, 156, 196],
    2: [179, 116, 81],
    3: [241, 149, 131],
    4: [81, 163, 148],
    5: [77, 174, 84],
    6: [108, 135, 75],
    7: [41, 49, 101],
    8: [79, 79, 76],
    9: [223, 52, 52],
    10: [89, 47, 95],
    11: [81, 109, 114],
    12: [233, 233, 229]
}


class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {i: cat for i, cat in enumerate(seg_classes.keys())}


def parse_args():
    parser = argparse.ArgumentParser('TMF Whole Scene Testing')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in testing')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device')
    parser.add_argument('--num_point', type=int, default=4096, help='point number per block')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment directory name under log/sem_seg/')
    parser.add_argument('--visual', action='store_true', default=False, help='save visualization results')
    parser.add_argument('--test_area', type=int, default=5, help='area for testing, option: 1-6')
    parser.add_argument('--num_votes', type=int, default=3, help='number of voting times')
    parser.add_argument('--num_classes', type=int, default=13, help='number of classes')
    parser.add_argument('--input_channels', type=int, default=9, help='input feature channels')
    parser.add_argument('--data_root', type=str, default='data/stanford_indoor3d/', help='dataset root')
    return parser.parse_args()


def add_vote(vote_label_pool, point_idx, pred_label, weight):
    """
    vote_label_pool: [num_points_in_scene, num_classes]
    point_idx: [B, N]
    pred_label: [B, N]
    weight: [B, N]
    """
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool


def main(args):
    def log_string(out_str):
        logger.info(out_str)
        print(out_str)


    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment_dir = Path('log/sem_seg') / args.log_dir
    visual_dir = experiment_dir / 'visual'
    visual_dir.mkdir(exist_ok=True)


    logger = logging.getLogger("TMF-Test")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(experiment_dir / 'eval.txt'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    log_string('PARAMETER ...')
    log_string(str(args))

    NUM_CLASSES = args.num_classes
    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point


    TEST_DATASET_WHOLE_SCENE = ScannetDatasetWholeScene(
        root=args.data_root,
        split='test',
        test_area=args.test_area,
        block_points=NUM_POINT
    )
    log_string("The number of test data is: %d" % len(TEST_DATASET_WHOLE_SCENE))


    log_files = os.listdir(experiment_dir / 'logs')
    assert len(log_files) > 0, f'No log file found in {experiment_dir / "logs"}'
    model_name = log_files[0].split('.')[0]
    log_string('Model name: %s' % model_name)

    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(
        num_classes=NUM_CLASSES,
        input_channels=args.input_channels
    ).to(device)

    checkpoint_path = experiment_dir / 'checkpoints' / 'best_model.pth'
    assert checkpoint_path.exists(), f'Checkpoint not found: {checkpoint_path}'
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    classifier.load_state_dict(checkpoint['model_state_dict'], strict=False)
    classifier.eval()


    with torch.no_grad():
        scene_id = TEST_DATASET_WHOLE_SCENE.file_list
        scene_id = [x[:-4] for x in scene_id]  # remove .npy
        num_batches = len(TEST_DATASET_WHOLE_SCENE)

        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

        log_string('---- EVALUATION WHOLE SCENE ----')

        for batch_idx in range(num_batches):
            log_string("Inference [%d/%d] %s ..." % (batch_idx + 1, num_batches, scene_id[batch_idx]))

            total_seen_class_tmp = [0 for _ in range(NUM_CLASSES)]
            total_correct_class_tmp = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class_tmp = [0 for _ in range(NUM_CLASSES)]

            if args.visual:
                fout = open(os.path.join(visual_dir, scene_id[batch_idx] + '_pred.obj'), 'w')
                fout_gt = open(os.path.join(visual_dir, scene_id[batch_idx] + '_gt.obj'), 'w')

            whole_scene_data = TEST_DATASET_WHOLE_SCENE.scene_points_list[batch_idx]     # [N,6]
            whole_scene_label = TEST_DATASET_WHOLE_SCENE.semantic_labels_list[batch_idx] # [N]

            vote_label_pool = np.zeros((whole_scene_label.shape[0], NUM_CLASSES))

            for _ in tqdm(range(args.num_votes), total=args.num_votes, smoothing=0.9):
                scene_data, scene_label, scene_smpw, scene_point_index = TEST_DATASET_WHOLE_SCENE[batch_idx]
                num_blocks = scene_data.shape[0]
                s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE

                batch_data = np.zeros((BATCH_SIZE, NUM_POINT, args.input_channels), dtype=np.float32)
                batch_label = np.zeros((BATCH_SIZE, NUM_POINT), dtype=np.int64)
                batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT), dtype=np.int64)
                batch_smpw = np.zeros((BATCH_SIZE, NUM_POINT), dtype=np.float32)

                for sbatch in range(s_batch_num):
                    start_idx = sbatch * BATCH_SIZE
                    end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
                    real_batch_size = end_idx - start_idx

                    batch_data[:] = 0
                    batch_label[:] = 0
                    batch_point_index[:] = 0
                    batch_smpw[:] = 0

                    batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, :, :args.input_channels]
                    batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
                    batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]
                    batch_smpw[0:real_batch_size, ...] = scene_smpw[start_idx:end_idx, ...]

                    torch_data = torch.from_numpy(batch_data).float().to(device)   # [B, N, C]
                    torch_data = torch_data.transpose(2, 1)                        # [B, C, N]

                    seg_pred, _ = classifier(torch_data)                           # [B, N, NUM_CLASSES]
                    batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()

                    vote_label_pool = add_vote(
                        vote_label_pool,
                        batch_point_index[0:real_batch_size, ...],
                        batch_pred_label[0:real_batch_size, ...],
                        batch_smpw[0:real_batch_size, ...]
                    )

            pred_label = np.argmax(vote_label_pool, 1)

            for l in range(NUM_CLASSES):
                total_seen_class_tmp[l] += np.sum(whole_scene_label == l)
                total_correct_class_tmp[l] += np.sum((pred_label == l) & (whole_scene_label == l))
                total_iou_deno_class_tmp[l] += np.sum((pred_label == l) | (whole_scene_label == l))

                total_seen_class[l] += total_seen_class_tmp[l]
                total_correct_class[l] += total_correct_class_tmp[l]
                total_iou_deno_class[l] += total_iou_deno_class_tmp[l]

            iou_map = np.array(total_correct_class_tmp) / (np.array(total_iou_deno_class_tmp, dtype=np.float64) + 1e-6)
            arr = np.array(total_seen_class_tmp)
            tmp_iou = np.mean(iou_map[arr != 0]) if np.sum(arr != 0) > 0 else 0.0
            log_string('Mean IoU of %s: %.4f' % (scene_id[batch_idx], tmp_iou))
            log_string('----------------------------')

            # save txt prediction
            filename = os.path.join(visual_dir, scene_id[batch_idx] + '.txt')
            with open(filename, 'w') as pl_save:
                for i in pred_label:
                    pl_save.write(str(int(i)) + '\n')

            # save visualization
            if args.visual:
                for i in range(whole_scene_label.shape[0]):
                    color = S3DIS_COLOR_MAP[int(pred_label[i])]
                    color_gt = S3DIS_COLOR_MAP[ int(whole_scene_label[i])]

                    fout.write(
                        'v %f %f %f %d %d %d\n' % (
                            whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2],
                            color[0], color[1], color[2]
                        )
                    )
                    fout_gt.write(
                        'v %f %f %f %d %d %d\n' % (
                            whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2],
                            color_gt[0], color_gt[1], color_gt[2]
                        )
                    )

                fout.close()
                fout_gt.close()

        IoU = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float64) + 1e-6)

        iou_per_class_str = '------- IoU --------\n'
        for l in range(NUM_CLASSES):
            iou_per_class_str += 'class %s, IoU: %.3f \n' % (
                seg_label_to_cat[l] + ' ' * max(1, 16 - len(seg_label_to_cat[l])),
                total_correct_class[l] / float(total_iou_deno_class[l] + 1e-6)
            )

        log_string(iou_per_class_str)
        log_string('eval point avg class IoU: %f' % np.mean(IoU))
        log_string('eval whole scene point avg class acc: %f' % (
            np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float64) + 1e-6))
        ))
        log_string('eval whole scene point accuracy: %f' % (
            np.sum(total_correct_class) / float(np.sum(total_seen_class) + 1e-6)
        ))

        print("Done!")


if __name__ == '__main__':
    args = parse_args()
    main(args)