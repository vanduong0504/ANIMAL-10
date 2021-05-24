<h1> CIFAR10 CLASSIFICATION </h1>

Try building some deep learning models from scratch with Pytorch.

<h2> Train </h2>

```
python main.py --models "VGG16" --phase "train"
```

<h2> Test </h2>

```
python main.py --models "VGG16" --phase "test"
```

<h2> Dataset </h2>

You can reimplement `Data.py` for your dataset. 

<h2> Performance of  Models </h2>

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
    <td>...</td>
  </tr>
  <tr>
    <td>VGG19</td>
    <td>20.2M</td>
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
    <td>21.3</td>
    <td>...</td>
  </tr>
  <tr>
    <td>Resnet50</td>
    <td>23.5M</td>
    <td>...</td>
  </tr>
  <tr>
    <td>Resnet152</td>
    <td>...</td>
    <td>...</td>
  </tr>
</tbody>
</table>