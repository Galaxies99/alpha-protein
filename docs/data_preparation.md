# Data Preparation

## Provided dataset
1. Download dataset `feature.zip` and `label.zip` from [google drive](https://drive.google.com/drive/folders/1rDsIOE8eAVL46tMMjZTsk94c8TVlLBUV?usp=sharing)
2. Unzip them into ` your_project/data` directory.
   
   Then, the `data` folder should have the following structure.

   ```
   data
   ├── feature
   |   ├── *.npy.gz
   |   └── *.npy.gz
   └── label
       ├── *.npy.gz
       └── *.npy.gz
   ```
3. Execute our data preprocessing python file to divide data files into three parts: 
   * Training data: 80%
   * Evaluation data: 10%
   * Testing data: 10%
   ```bash
   cd utils
   python preprocessing.py
   ```

   Then, the `data` folder should have the following structure.

   ```
   data
   ├── feature
   |   ├── *.npy.gz
   |   └── *.npy.gz
   ├── label
   |   ├── *.npy.gz
   |   └── *.npy.gz
   ├── test
   |   ├── feature
   |   |   ├── *.npy.gz
   |   |   └── *.npy.gz
   |   └── label
   |       ├── *.npy.gz
   |       └── *.npy.gz
   ├── train
   |   ├── feature
   |   |   └── *.npy.gz
   |   └── label
   |       └── *.npy.gz
   └── val
       ├── feature
       |   └── *.npy.gz
       └── label
           └── *.npy.gz
   ```

## Prepared by yourself
If you want to prepare for the data yourselves, you should package your features and labels data into `*.npy.gz` files respectively, which means
a numpy array should be output as a `.npy` file and zipped into `.gz` file.

**Feature**: Precision matrix of Multiple Sequence Alignments (MSAs)
, a `L*L*441` numpy array.

**Label**: Protein contact map with ten classes, a `L*L*10` numpy array.



