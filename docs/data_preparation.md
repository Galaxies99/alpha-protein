# Data Preparation

## Provided dataset

1. Download our provided dataset `feature.zip` and `label.zip` from [Google Drive](https://drive.google.com/drive/folders/1rDsIOE8eAVL46tMMjZTsk94c8TVlLBUV?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1XJ5o8TTQT7HFG4w3XcMR7w) (Extract Code: fcjv).

2. Unzip them into the `data` directory. Then, the `data` folder should have the following structure.

   ```
   data
   ├── feature
   |   ├── *.npy.gz
   |   └── *.npy.gz
   └── label
       ├── *.npy
       └── *.npy
   ```

3. Execute our data preprocessing python file to divide data files into three parts:
   * Training data: 80%
   * Evaluation data: 10%
   * Testing data: 10%
  
   ```bash
   python utils/preprocessing.py
   ```

   After that, the `data` folder should have the following structure.

   ```
   data
   ├── feature
   |   ├── *.npy.gz
   |   └── *.npy.gz
   ├── label
   |   ├── *.npy
   |   └── *.npy
   ├── test
   |   ├── feature
   |   |   ├── *.npy.gz
   |   |   └── *.npy.gz
   |   └── label
   |       ├── *.npy
   |       └── *.npy
   ├── train
   |   ├── feature
   |   |   └── *.npy.gz
   |   └── label
   |       └── *.npy
   └── val
       ├── feature
       |   └── *.npy.gz
       └── label
           └── *.npy
   ```

## Prepared by yourself

If you want to prepare for the data yourselves, you should package your features and labels data into `*.npy.gz` files and `.npy` files respectively, and put them under the data folder as illustrated below.

```
data
├── feature
|   ├── *.npy.gz
|   └── *.npy.gz
└── label
   ├── *.npy
   └── *.npy
```

The provided configuration of our model will take an LxLx441 matrix as an input and output an LxLx10 contact-map, and you can change the input channel and output channel in the configuration files to satisfy your input/output channels. For details about the configuration file, see [docs/configurations.md](configurations.md)
