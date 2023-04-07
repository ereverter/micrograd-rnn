import random
from engine import Value
from nn import Module

class RNNNeuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w_xh = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.w_hh = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.h = Value(0)
        self.nonlin = nonlin

    def __call__(self, x, h_prev):
        h = sum(wi * xi for wi, xi in zip(self.w_xh, x)) + sum(wj * hj for wj, hj in zip(self.w_hh, h_prev)) + self.b
        h = h.tanh() if self.nonlin else h
        self.h = h
        return h
    
    def parameters(self):
        return self.w_xh + self.w_hh + [self.b]

    def __repr__(self):
        return f"{'tanh' if self.nonlin else 'Linear'}RNNNeuron({len(self.w_xh)})"

class RNNLayer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neuron = RNNNeuron(nin, **kwargs)
        self.nout = nout

    def __call__(self, x, h_prev):
        h_new = []
        for i in range(len(x)):
            h = self.neuron(x[i], h_prev)
            h_new.append([h])
            h_prev = [h]
        return h_new, h_new

    def parameters(self):
        return self.neuron.parameters()

    def __repr__(self):
        return f"RNNLayer of {self.nout} [{self.neuron}]"

class RNN(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [RNNLayer(sz[i], sz[i + 1], nonlin=i != len(nouts) - 1) for i in range(len(nouts))]

    def __call__(self, x, h_prevs):
        h_new_all = []
        for layer_idx, layer in enumerate(self.layers):
            print('layer_idx', layer_idx)
            h_prev = h_prevs[layer_idx]
            x, h_new = layer(x, h_prev)
            h_new_all.append(h_new)
        return x, h_new_all

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"RNN of [{', '.join(str(layer) for layer in self.layers)}]"
