<h1 align="center"> IMAGE CLASSIFICATION </h1>

[![Pytorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)


<h2> :pencil: About the project </h2>

This project aims to classify image in **CIFAR10** dataset with deep learning models. I try to implement it from scratch. 
![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2> :computer: Train/Test </h2>
<h4> Train </h4>

```
python main.py --models "VGG16" --phase "train"
```
<h4> Test </h4>

```
python main.py --models "VGG16" --phase "test"
```
![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2> :floppy_disk: Dataset </h2>

I use **CIFAR10** as example dataset for my implementation. You can reimplement `data.py` for your custom dataset. 
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
    <td>14.9M</td>
    <td>92.09</td>
  </tr>
  <tr>
    <td>VGG19</td>
    <td>20.2M</td>
    <td>91.74</td>
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
    <td>...</td>
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
