## Multiclass Segmentation with UNet
Multiclass segmentation using [EVIMO2](https://better-flow.github.io/evimo/download_evimo_2.html#sfm) dataset with UNet.
![myplot.png](utils%2Fmyplot.png)

## Requirements
* Python 3.8
* PyTorch
* CUDA 11.7

## Usage
```
pip install -r requirements.txt
```

The recommended directory structure for the entire project is as follows:
```
.
├─ multiclass-segmentation-with-UNet
├── evimo
│   └── flea3_7
│       └── sfm
│           ├── train
│           └── valid
│   └── ...
├── model # saving the best models during training
├── utils
├── main.py
└── pred.py
```

Convert the original npy files to images
```
python data_extraction.py
```

Train the model
```
python main.py
```

Evaluate the model and visualize outputs
```
python pred.py
```