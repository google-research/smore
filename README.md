# SMORE: Knowledge Graph Completion and Multi-hop Reasoning in Massive Knowledge Graphs

SMORE is a a versatile framework that scales multi-hop query embeddings over KGs. SMORE can easily train query embeddings on Freebase KG with more than 86M nodes and 338M edges on a single machine. For more details, please refer to our [paper](https://icml21ssl.github.io/pages/files/Scaling_poster.pdf).

## Overview
![](https://github.com/Hanjun-Dai/multihop_kg/blob/license/assets/pipeline.png?raw=true)
SMORE designs an optimized pipeline with the following features.
- [x] Multi-GPU Training
- [x] Bidirectional Online Query Sampling

## Installation

First clone the repository, and install the package dependency required in the `requirements.txt`. 

Then navigate to the root folder of the project and do 

    git submodule update --init
    pip install -e .

## Models

SMORE supports six different singel / multi-hop reasoning methods on KG.
- [x] [BetaE](https://arxiv.org/abs/2010.11465)
- [x] [Query2box](https://arxiv.org/abs/2002.05969)
- [x] [GQE](https://arxiv.org/abs/1806.01445)
- [x] [RotatE](https://arxiv.org/abs/1902.10197)
- [x] [ComplEx](https://arxiv.org/abs/1606.06357)
- [x] [DistMult](https://arxiv.org/abs/1412.6575)

## Examples

Please see the example script of each methods under `smore/training` folder. We provide example scripts of the six query emebddings on six KGs.

## Contributing

We welcome pull request, please check [CONTRIBUTING.md](https://github.com/Hanjun-Dai/multihop_kg/blob/license/CONTRIBUTING.md) for more details.

## Citations

If you use this repo, please cite the following paper.

```
@article{
 ren2020scaling,
 title={Scaling up Logical Query Embeddings on Knowledge Graphs},
 author={Ren, Hongyu and Dai, Hanjun and Dai, Bo and Chen, Xinyun and Zhou, Denny and Leskovec, Jure and Schuurmans, Dale},
 year={2021}
}
```

## License
SMORE is licensed under the Apache License, Version 2.0.

This is not an officially supported Google product.

Contact hyren@cs.stanford.edu and hadai@google.com for questions about the repo.
