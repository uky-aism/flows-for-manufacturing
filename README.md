# Flows for Manufacturing

This code is for the MSEC 2023 paper "Normalizing Flows for Intelligent Manufacturing."

## Setup
```
python -m pip install -r requirements.txt
```

## Image Generation Experiments
```
python -m flows4manufacturing.image_generation.generation \
    --data ../path/to/KolektorSDD2/train \
    --epochs 5000 \
    --lr 0.001 \
    --save flow.pt \
    --save-ae autoencoder.pt
```

## Parameter Estimation Experiments
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