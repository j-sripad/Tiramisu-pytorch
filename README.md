

This repository contains the  implementation of "The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation" in pytorch.

#### Paper

* [View the Preprint](https://arxiv.org/abs/1611.09326)

#### Note
* The decoder part of this implementation is bit different from that of the paper. 

#### Architecture
![Alt text](tiramisu.png?raw=true "Title")


#### Usage
<pre><code>
from tiramisu import Tiramisu_Segmentation <br>
net = Tiramisu_Segmentation(layer_tiramisu=103,nclasses=1,input_features=1,growth_rate=16)
<br>
"""
Arguments : 
layer_tiramisu - 57, 47 or 103
input_features - input image channels
nclasses - number of classes
growth_rate - growth rate (filters  to begin with for convolution - generally 16)
"""
#### Dataset
* [Person Detection Data](https://supervise.ly/)


<img src="https://cloud.githubusercontent.com/assets/4307137/10105283/251b6868-63ae-11e5-9918-b789d9d682ec.png" width="30%"></img> <img src="https://cloud.githubusercontent.com/assets/4307137/10105290/2a183f3a-63ae-11e5-9380-50d9f6d8afd6.png" width="30%"></img> <img src="https://cloud.githubusercontent.com/assets/4307137/10105284/26aa7ad4-63ae-11e5-88b7-bc523a095c9f.png" width="30%"></img>




