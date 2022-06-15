<h1 align = "center">Siamese Network</h1>

## Organization of this Repo

```
.
├── README.md
├── logs
│   ├── 3-shot-aug.out
│   ├── 3-shot.out
│   ├── 5-shot-aug.out
│   ├── 5-shot.out
│   ├── sia_3_shot_aug_epoch_loss.png
│   ├── sia_3_shot_epoch_loss.png
│   ├── sia_5_shot_aug_epoch_loss.png
│   └── sia_5_shot_epoch_loss.png
├── nets
│   ├── __init__.py
│   ├── siamese.py
│   └── vgg.py
├── oracle_fs: The dataset locate at.
├── predict.py
├── requirements.txt
├── siamese.py
├── train.py
└── utils
    ├── __init__.py
    ├── callbacks.py
    ├── dataloader.py
    ├── utils.py
    ├── utils_aug.py
    ├── utils_fit.py
    └── utils_unmod.py
```

## Usage

1. Put the Oracle-FS dataset in this repo.
2. Modify the folder path in `train.py`.
3. Run `python train.py`

## Reference

https://github.com/bubbliiiing/Siamese-pytorch