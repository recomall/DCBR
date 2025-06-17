# DCBR (AAAI'25)
PyTorch implementation for DCBR proposed in the following paper:
 >**Disentangled Contrastive Bundle Recommendation with Conditional Diffusion**  
 >Jiuqiang Li  
 >In *AAAI 2025*  
 >[Paper](https://doi.org/10.1609/aaai.v39i11.33314)

## Overview
<p>
<img src="./assets/DCBR.png" width="800">
</p>

## Environment
We implement our code in the following environment.

- OS: Ubuntu 20.04
- GPU: NVIDIA RTX 3090(24GB) * 1
- CPU: 14 vCPU Intel(R) Xeon(R) Platinum 8362 CPU @ 2.80GHz
- Python 3.8.10
- torch==1.11.0+cu113
- numpy==1.22.4
- PyYAML==6.0.2
- scipy==1.10.1
- tqdm==4.61.2

Run the following command to install the dependencies.
```bash
pip install -r requirements.txt
```

## Dataset
The datasets utilized in the experiments include $MealRec_H^+$, $MealRec_L^+$, and iFashion. $MealRec^+$ and iFashion have been published in the [MealRecPlus](https://github.com/WUT-IDEA/MealRecPlus/) and [CrossCBR](https://github.com/mysbupt/CrossCBR) codebases, respectively.

## Training
The command to train DCBR on the $MealRec_H^+$ / $MealRec_L^+$ / iFashion dataset is as follows.
```bash
python train.py -m DCBR -d {dataset_name}
```
  
Please set the hyperparameters in the `configs/models/DCBR.yaml` file.

## Evaluation
The command to evaluate the performance of the pretrained checkpoints on the specified dataset is as follows.
```bash
python test.py -m DCBR -d {dataset_name} -c {checkpoint_path}
```
In `test.py`, two evaluation methods are provided. Specifically:

- `quick_test()`: Directly calculates metrics using the pre-trained embeddings from the checkpoints.
- `test()`: Infers to obtain embeddings before calculating metrics.
## Reproducibility
We report the best configuration, training logs, and corresponding checkpoints of DCBR to reproduce the results in Table 2 of our paper.
<table>
  <tr>
    <th>Dataset</th>
    <th colspan="3">Download</th>
  </tr>
  <tr>
    <td align="center">MealRec<sub>H</sub><sup>+</sup></td>
    <td><a href="https://drive.google.com/file/d/1C6kWrU1t7Xq6DSgB1jqyW6P4ioX5sl7t/view">conf</a></td>
    <td><a href="https://drive.google.com/file/d/1H26X_ADYM7SVmBrTQ0bGCVkbNpnO21do/view">log</a></td>
    <td><a href="https://drive.google.com/file/d/1QeaaHTPkO-g8yd73ZjBtbYdjYVswCE-0/view">checkpoint</a></td>
  </tr>
  <tr>
    <td align="center">MealRec<sub>L</sub><sup>+</sup></td>
    <td><a href="https://drive.google.com/file/d/1MrrL24N6XrSdrLdHEfFOvq5qRm64SDQ8/view">conf</a></td>
    <td><a href="https://drive.google.com/file/d/1rRFtMWXguDCPeLBdxVylwTLPPjsDk5-x/view">log</a></td>
    <td><a href="https://drive.google.com/file/d/1HlAFDY5RTmx9RatkirTYGZSV5m_ChgsL/view">checkpoint</a></td>
  </tr>
  <tr>
    <td align="center">iFashion</td>
    <td><a href="https://drive.google.com/file/d/1DJMRDDWSApmgo8dIvPm-lL47qu9nnMPe/view">conf</a></td>
    <td><a href="https://drive.google.com/file/d/1mN6sRN81NkRw5AFX6nyD_VyTLxIKNRlW/view">log</a></td>
    <td><a href="https://drive.google.com/file/d/1F3wN8EGcs0f5AaR92-pk25va9M9Qejn5/view">checkpoint</a></td>
  </tr>
</table>

## Citation
If you find this work is helpful to your research, please kindly cite the following paper.
```bibtex
@inproceedings{li2025disentangled,
  title={Disentangled Contrastive Bundle Recommendation with Conditional Diffusion},
  author={Li, Jiuqiang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={11},
  pages={12067--12075},
  year={2025}
}
```

## Acknowledgement
​​This repository is based on [CrossCBR](https://github.com/mysbupt/CrossCBR), [MultiCBR](https://github.com/HappyPointer/MultiCBR), [DiffRec](https://github.com/YiyanXu/DiffRec) and [CoHeat](https://github.com/snudatalab/CoHeat). Thanks for their work.