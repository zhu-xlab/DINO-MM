# DINO-MM
[Self-supervised vision transformers for joint SAR-optical representation learning, IGARSS 2022](https://arxiv.org/abs/2204.05381)


*Note: Codes may not run out-of-the-box, a cleaner version to be organized.*



### training and linear evaluation
run `sbatch scripts/slurm/sar_optical/srun_xxx.sh`


### pre-trained model

DINO-MM with ViT-S/8, input 14 bands: [checkpoint](https://huggingface.co/wangyi111/dino-mm/resolve/main/B14_vits8_dinomm_ep99.pth), [training log](checkpoints/pretrain_log.txt)


## Further reads
"Self-supervised Learning in Remote Sensing: A Review". [paper](https://arxiv.org/abs/2206.13188) | [Repository](https://github.com/zhu-xlab/SSL4EO-Review)

"SSL4EO-S12: A Large-Scale Multi-Modal, Multi-Temporal Dataset for Self-Supervised Learning in Earth Observation". [Paper](https://arxiv.org/abs/2211.07044) | [Repository](https://github.com/zhu-xlab/SSL4EO-S12)

"DeCUR: decoupling common & unique representations for multimodal self-supervision". [paper](https://arxiv.org/abs/2309.05300) | [Repository](https://github.com/zhu-xlab/DeCUR)
