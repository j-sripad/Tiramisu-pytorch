from typing import Tuple, Dict
import torch.nn as nn
import torch.nn.functional as F
import torch

class Dense_Block(nn.Module):
    def __init__(self,input_features : int, output_features : int,layer_count : int =4  , padding : int = 0, kernel : tuple = (3,3),stride : tuple = (1,1), dropout : float = 0.2 )->None:
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.layer_count = layer_count
        self.padding = padding
        self.kernel = kernel
        self.stride = stride
        self.in_loop_features = 0
        self.dropout = dropout

        #layers
        self.dropout = nn.Dropout(p=0.2)
        self.conv_layers_list = nn.ModuleList([])
        self.batch_norm_list = nn.ModuleList([])
        for i in range(0,self.layer_count-1):
            if i == 0:
                self.batch_norm_list.append(nn.BatchNorm2d(self.input_features,affine=False))

                self.conv_layers_list.append(nn.Conv2d(in_channels=self.input_features,out_channels=self.output_features,kernel_size=self.kernel,stride = self.stride,padding=self.padding))
                self.in_loop_features = self.input_features + self.output_features
            
            self.batch_norm_list.append(nn.BatchNorm2d(self.in_loop_features,affine=False))
            self.conv_layers_list.append(nn.Conv2d(in_channels=self.in_loop_features,out_channels=self.output_features,kernel_size=self.kernel,stride = self.stride,padding=self.padding))
            if i == self.layer_count-2:
                continue
            self.in_loop_features = self.in_loop_features + self.output_features

        self.final_batch_norm = nn.BatchNorm2d(self.in_loop_features,affine=False)
        self.final_conv = nn.Conv2d(in_channels=self.in_loop_features,out_channels=self.output_features,kernel_size=self.kernel,stride = self.stride,padding=self.padding)
        
    
    def forward(self,x : torch.Tensor)->torch.Tensor:
        layer_output_list = []
        for i in range(0,self.layer_count-1):
            
            x_dummy = self.batch_norm_list[i](x)
            x_dummy = F.relu(x_dummy)
            x_dummy = self.conv_layers_list[i](x_dummy)
            x_dummy = self.dropout(x_dummy)
            layer_output_list.append(x_dummy)
            x = torch.cat([x_dummy,x],axis = 1)
        x = self.final_batch_norm(x)
        x = self.final_conv(x)
        temp = torch.cat(layer_output_list,axis=1)
        x = torch.cat([x,temp],axis=1)
        return x

class Transition_Down(nn.Module):
    def __init__(self,features : int,dropout : int = 0.2,padding : int = 0):
        super().__init__()
        self.features = features
        self.dropout = dropout
        self.padding = padding


        self.dropout_en = nn.Dropout(p=self.dropout)
        self.batch_norm =nn.BatchNorm2d(self.features,affine=False)
        self.conv = nn.Conv2d(in_channels=self.features,out_channels=self.features,kernel_size=(1,1),stride=(1,1),padding = self.padding)
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2),stride = (2,2),padding=self.padding)

    def forward(self,x : torch.Tensor)->torch.Tensor:
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.conv(x)
        x = self.dropout_en(x)
        x = self.maxpool(x)

        return x

class Transition_Up(nn.Module):
    def __init__(self,features : int, kernel : tuple = (2,2),padding : int = 0):
        super().__init__()
        self.features = features
        self.kernel = kernel
        self.padding = padding
        

       
        self.conv_inv = nn.ConvTranspose2d(in_channels=self.features,out_channels=self.features,kernel_size=self.kernel,stride=(2,2),padding = self.padding)

    def forward(self,x : torch.Tensor)->torch.Tensor:
       
        return self.conv_inv(x)


class Tiramisu_Segmentation(nn.Module):
    def __init__(self,layer_tiramisu : int = 57,dropout : int = 0.2,padding : int = 1, input_features : int = 1 , growth_rate : int = 16, first_conv_output_features : int = 48 , nclasses : int = 1):

        super().__init__()
        self.layer_tiramisu = layer_tiramisu
        if self.layer_tiramisu == 103:
            self.layer_list = [4,5,7,10,12,15,12,10,7,5,4]
        elif self.layer_tiramisu == 67:
            self.layer_list = [5,5,5,5,5,5,5,5,5,5,5]

        else:
            self.layer_list = [4,4,4,4,4,4,4,4,4,4,4]
        self.nclasses = nclasses
        self.dropout = dropout
        self.padding =padding
        self.input_features = input_features
        self.growth_rate = growth_rate
        self.first_conv_output_features = first_conv_output_features
        self.filters_per_layer_encoder = []
        #layers
        self.dense_conv_block_list = nn.ModuleList([])
        self.td_list = nn.ModuleList([])
        self.conv_initial = nn.Conv2d(in_channels=self.input_features,out_channels=self.first_conv_output_features,kernel_size=(3,3),stride=(1,1),padding=1)
        temp = 0
        for i in range(0,5):
            
            if i == 0:
                
                self.dense_conv_block_list.append(Dense_Block(input_features=self.first_conv_output_features,output_features = self.growth_rate,layer_count = self.layer_list[i],padding=self.padding))
                self.td_list.append(Transition_Down(features=(self.growth_rate*(self.layer_list[i]))+self.first_conv_output_features))
                temp = self.first_conv_output_features + (self.growth_rate* self.layer_list[i]) 
            else:

                self.dense_conv_block_list.append(Dense_Block(input_features=temp,output_features = self.growth_rate,layer_count = self.layer_list[i],padding=self.padding))
                self.td_list.append(Transition_Down(features=(self.growth_rate*(self.layer_list[i])+temp)))
                temp = temp + (self.growth_rate*(self.layer_list[i]) ) 
            self.filters_per_layer_encoder.append(temp)

        self.mid_dense = Dense_Block(input_features=temp,output_features = self.growth_rate,layer_count = self.layer_list[5],padding=self.padding)


        self.input_to_decoder_filters = temp + self.growth_rate * self.layer_list[5]
        self.dense_conv_block_list_decoder = nn.ModuleList([])
        self.tu_list = nn.ModuleList([])

        controller = 2
        for i in range(6,len(self.layer_list)):
            self.tu_list.append(Transition_Up(features=  self.input_to_decoder_filters))
            
            self.dense_conv_block_list_decoder.append(Dense_Block(input_features=self.input_to_decoder_filters,output_features=self.growth_rate,layer_count=self.layer_list[i],padding = 1))
            self.input_to_decoder_filters =  self.growth_rate * self.layer_list[i] + self.filters_per_layer_encoder[i-controller]
       

            # print(self.input_to_decoder_filters)
            
            controller+=2

            
        self.conv_final = nn.Conv2d(in_channels=self.input_to_decoder_filters,out_channels= self.nclasses,
    kernel_size = (1,1), stride=(1,1))

        self.softmax = nn.Softmax(dim=self.nclasses)
        self.sigmoid = nn.Sigmoid()



    def forward(self,x : torch.Tensor)->torch.Tensor:
        x_skip_layers =  []
        x = self.conv_initial(x)
        x_og = x
        for i in range(len(self.dense_conv_block_list)):
            x = self.dense_conv_block_list[i](x)
            x = torch.cat([x_og,x],axis=1)
            x_skip_layers.append(x)
            # print(x.shape)
            x = self.td_list[i](x)
            x_og = x
        
        x = self.mid_dense(x)
        x = torch.cat([x_og,x],axis=1)
        for i in range(len(self.dense_conv_block_list_decoder)):
            x = self.tu_list[i](x)
            x = self.dense_conv_block_list_decoder[i](x)
            x = torch.cat([x,x_skip_layers[-(i+1)]],axis=1)
        x = self.conv_final(x)
#         print(x.shape)
#         print(x.type)
        if self.nclasses > 1:
            x =  self.softmax(x)
        else:
            x =  self.sigmoid(x)
        
    

        return x


