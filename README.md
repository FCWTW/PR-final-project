# Pattern Recognition Final Project

## Code source

https://github.com/NVlabs/SegFormer

https://github.com/speedinghzl/CCNet/tree/pure-python?tab=readme-ov-file

## Environment

Python = 3.9、
CUDA = 11.8、
pytorch = 2.4.0、
mmcv = 2.1.0、
mmengine = 0.10.5、
mmseg = 1.2.2

```bash
pip install -U openmim
mim install "mmcv==2.1.0"
mim install mmengine
mim install mmsegmentation

pip install ftfy
pip install regex
```

## Dataset
https://www.cityscapes-dataset.com/

## Result showcase
https://youtu.be/XEacWkDK8iE

## Reference

[1] Zilong Huang, Xinggang Wang, Yunchao Wei, Lichao Huang, Humphrey Shi, Wenyu Liu, and Thomas S. Huang. Ccnet: Criss-cross attention for semantic segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence, pages 1–1, 2020.

[2] Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M. Alvarez, and Ping Luo. Segformer: Simple and efficient design for semantic segmentation with transformers. arXiv preprint arXiv:2105.15203, 2021

[3] M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele. The cityscapes dataset for semantic urban scene understanding. In Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 3213–3223, 2016.

[4] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 248–255, 2009.

[5] Zilong Huang, Yunchao Wei, Xinggang Wang, and Wenyu Liu. A pytorch semantic segmentation toolbox. https://github.com/speedinghzl/pytorch-segmentation-toolbox, 2018.

[6] MMSegmentation Contributors. Mmsegmentation: Openmmlab semantic segmentation toolbox and benchmark. https://github.com/open-mmlab/mmsegmentation, 2020. Accessed: 2025-01-02.
