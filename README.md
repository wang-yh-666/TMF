TMF: Topology-Aware Multi-Information Fusion for Object Recognition

This repository provides the core implementation of the TMF (Topology-Aware Multi-Information Fusion) model.


Training


python train.py --model tmf --batch_size 32 --npoint 4096 --num_classes 13 --input_channels 9
```

Testing


python test.py --log_dir <experiment_folder> --num_point 4096 --num_classes 13
```

---

Code Availability

The code is publicly available:

GitHub: https://github.com/wang-yh-666/TMF