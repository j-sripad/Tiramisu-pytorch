![Ground Truth_1](https://user-images.githubusercontent.com/39008190/117753962-057dad80-b237-11eb-9b02-1b2ec5beabe9.png)
![Ground Truth_0](https://user-images.githubusercontent.com/39008190/117753918-f696fb00-b236-11eb-87fe-3e43400524a8.png)


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
</code></pre>


#### Dataset
* [Person Detection Data](https://supervise.ly/)


<img src="https://github.com/j-sripad/Tiramisu-pytorch/blob/main/misc/Image_0.png" width="33%"></img> 
<img src="https://github.com/j-sripad/Tiramisu-pytorch/blob/main/misc/Ground%20Truth_0.png" width="33%"></img> 
<img src="https://github.com/j-sripad/Tiramisu-pytorch/blob/main/misc/predicted_0.png" width="33%"></img>




