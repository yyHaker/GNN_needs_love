Graph Convolutional Networks in PyTorch
====

PyTorch implementation of Graph Convolutional Networks (GCNs) for semi-supervised classification [1].

For a high-level introduction to GCNs, see:

Thomas Kipf, [Graph Convolutional Networks](http://tkipf.github.io/graph-convolutional-networks/) (2016)


![Graph Convolutional Networks](files/figure.png)

Note: There are subtle differences between the TensorFlow implementation in https://github.com/tkipf/gcn and this PyTorch re-implementation. This re-implementation serves as a proof of concept and is not intended for reproduction of the results reported in [1].

This implementation makes use of the Cora dataset from [2].

For Graph Convolutional Networks (GCNs), here we have two kinds of implementation, one is use pure numpy and PyTorch, the other is use torch-geometric and PyTorch.

## Requirements

  * Python 3.6
  * PyTorch 1.2.0
  * torch-cluster 1.4.5
  * torch-scatter 1.4.0
  * torch-sparse 0.4.3
  * torch-geometric 1.3.2

## Usage

```python train_gcn.py```

```python train_Kipfgcn.py```

## Result(accuracy here)
dataset | train | dev | test
---|---|---|---|
Cora| 0.986| 0.81 | 0.81 |
citeseer| 0.983| 0.728 | 0.714 |
pubmed| 1.0| 0.798 | 0.794 |


## References

[1] [Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks, 2016](https://arxiv.org/abs/1609.02907)

[2] [Sen et al., Collective Classification in Network Data, AI Magazine 2008](http://linqs.cs.umd.edu/projects/projects/lbc/)
