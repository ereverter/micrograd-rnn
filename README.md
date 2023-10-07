# Extension of micrograd for RNNs

Simple extension of [micrograd](https://github.com/karpathy/micrograd) to consider a recurrent neural network (RNN) architecture. The RNN definition is found in [extension.py](extension.py). With collaboration of [@amartorell98](https://github.com/amartorell98).

The only modification made to the original code is the inclusion of the `extension.py` script. Within the `playground.ipynb` file, you can locate the functions responsible for calculating the BPTT (Backpropagation Through Time) loss and conducting the training process. The data used for training is generated artificially. By adjusting parameters such as the number of epochs, gradient clipping value, learning rate value, and its decay, you can fine-tune the training process.
