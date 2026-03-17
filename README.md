# TMF: Topology-Aware Multi-Information Fusion for Object Recognition

This repository provides the core implementation of the TMF (Topology-Aware Multi-Information Fusion) model.

---

## Training

```bash
python train.py --model tmf --batch_size 8 --npoint 2048 --num_classes 13 --input_channels 9
```

---

## Testing

```bash
python test.py --log_dir <experiment_folder> --num_point 2048 --num_classes 13
```

---

## Notes

This repository provides a cleaned implementation of the TMF model.

Due to differences in hardware environments, training configurations, and random initialization, the reproduced results may vary from those reported in the paper.
