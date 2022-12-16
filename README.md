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

## Citation
If you find this code helpful in your reesearch, please cite it.
This citation will be updated to reflect the conference publication
once the proceedings are published.
```
@misc{russell2022,
    author = {Matthew Russell},
    title  = {Flows for Manufacturing},
    year   = {2022},
    url    = {https://github.com/uky-aism/flows-for-manufacturing} 
}
```

## Contact Information
| Name |  GitHub | Email |
|-|-|-|
|Matthew Russell|@mbr4477|matthew.russell@uky.edu|
