<h1 align="center"> IMAGE CLASSIFICATION </h1>

[![Pytorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)


<h2> :pencil: About the project </h2>

This project aims to classify images in [**ANIMAL-10**](https://www.kaggle.com/alessiocorrado99/animals10) dataset with deep learning models. I try to implement it from scratch. 
![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2> :computer: Train/Test </h2>
<h4> Train </h4>

```
python main.py --models "VGG16" --phase "train" --dataroot "path/to/dataroot" 
```
<h4> Test </h4>

```
python main.py --models "VGG16" --phase "test" --dataroot "path/to/dataroot" --load_path "path/to/weight"
```
<h4> Image Testing </h4>

```
python main.py --models "VGG16" --phase "test" --load_path "path/to/weight" --image_path "path/to/image" 
```

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2> :floppy_disk: Dataset </h2>

For custom dataset, follow this structure:

    📂 your dataset
    |─── train
    |      |─────class-1
    |      |        +-- *.jpg
    |      |─────class-2
    |      |        +-- *.jpg
    |─── test
    |      |─────class-1
    |      |        +-- *.jpg
    |      |─────class-2
    |      |        +-- *.jpg

    # For example
    📂 Dog-and-Cat
    |─── train
    |      |─────class-1
    |      |        +-- 001.jpg
    |      |        +-- 002.jpg
    |      |─────class-2
    |      |        +-- 003.jpg
    |      |        +-- 004.jpg
    |─── test
    |      |─────class-1
    |      |        +-- 001.jpg
    |      |        +-- 002.jpg
    |      |─────class-2
    |      |        +-- 003.jpg
    |      |        +-- 004.jpg



![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)


<h2> :scroll: Configuration </h2>

My configuration for training phase:
<ul>
  <li> <strong>Batch size</strong> : 64</li>
  <li> <strong>Epoch</strong> : 50</li>
  <li> <strong>Learing rate</strong> : 0.001</li>
</ul>


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2> :chart_with_upwards_trend:Performance of Models </h2>

<table style="undefined;table-layout: fixed; width: 900px">

<thead>
  <tr>
    <th rowspan="2">Model</th>
    <th rowspan="2">Method</th>
    <th rowspan="2">#Parameters</th>
    <th rowspan="2">Accuracy</th>
  </tr>
</thead>
<tbody>
  
  <tr>
    <td rowspan="2">VGG</td>
    <td>VGG16</td>
    <td>134.3M</td>
    <td>...</td>
  </tr>
  <tr>
    <td>VGG19</td>
    <td>139.6M</td>
    <td>...</td>
  </tr>
  
  <tr>
    <td rowspan="4">Resnet</td>
    <td>Resnet18</td>
    <td>11.2M</td>
    <td>...</td>
  </tr>
  <tr>
    <td>Resnet34</td>
    <td>21.3M</td>
    <td>...</td>
  </tr>
  <tr>
    <td>Resnet50</td>
    <td>23.5M</td>
    <td>.</td>
  </tr>
  <tr>
    <td>Resnet152</td>
    <td>58.1M</td>
    <td>...</td>
  </tr>
  
   <tr>
    <td rowspan="1">...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
  </tr>
  
</tbody>
</table>
