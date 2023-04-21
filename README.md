# Flows for Manufacturing

This code is for the MSEC 2023 paper "Normalizing Flows for Intelligent Manufacturing."

## Setup
```
git clone https://github.com/uky-aism/flows-for-manufacturing.git
python -m pip install -U pip
python -m pip install -e ./flows-for-manufacturing
```

## Image Generation Experiments
Dataset: https://www.vicos.si/resources/kolektorsdd2/
```
python -m flows4manufacturing.image_generation.generation \
    --data ../path/to/KolektorSDD2/train \
    --epochs 5000 \
    --lr 0.001 \
    --save flow.pt \
    --save-ae autoencoder.pt
```

## Milling Parameter Estimation Experiments
Dataset: "3. Milling" from https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository
```
python -m flows4manufacturing.parameter_estimation.estimate \
    --epochs 1000 \
    --lr 0.001 \
    --blocks 8 \
    --context 32 \
    --gru 2 \
    --hidden 32 \
    --data path/to/mill.mat \
    --noise-power 0.002 \
    --train 12000 \
    --val 3000 \
    --seed 0 \
    --out bayesflow00.pt
```

## Motor Anomaly Detection Experiments

> Data not available publicly at this time

```
python -m flows4manufacturing.anomaly_detection.main \
    --epochs 5000   \
    --hidden 256    \
    --lr 1e-5       \
    --blocks 16     \
    --seed 0        \
    --amp
```

Autoencoder and Deep SVDD experiments can be run using
`flows4manufacturing.anomaly_detection.autoencoder`
and `flows4manufacturing.anomaly_detection.deep_svdd`,
respectively.

## Citation
If you find this code helpful in your research, please cite it.
This citation will be updated to reflect the conference/journal publication when available.
```
@misc{russell2023,
    author = {Matthew Russell and Peng Wang},
    title  = {Normalizing Flows for Intelligent Manufacturing},
    year   = {2023},
    url    = {https://github.com/uky-aism/flows-for-manufacturing} 
}
```

## Contact Information
| Name |  GitHub | Email |
|-|-|-|
|Matthew Russell|@mbr4477|matthew.russell@uky.edu|
