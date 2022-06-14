#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py --dir=data/oracle_1shot --name=oracle_1shot --batch-size=256 --max-epoch=200
# CUDA_VISIBLE_DEVICES=0 python main.py --dir=data/oracle_3shot --name=oracle_3shot --batch-size=256 --max-epoch=200
# CUDA_VISIBLE_DEVICES=0 python main.py --dir=data/oracle_5shot --name=oracle_5shot --batch-size=256 --max-epoch=200
