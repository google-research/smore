# SMORE: Knowledge Graph Completion and Multi-hop Reasoning in Massive Knowledge Graphs

SMORE is a a versatile framework that scales multi-hop query embeddings over KGs. SMORE can easily train query embeddings on Freebase KG with more than 86M nodes and 338M edges on a single machine. For more details, please refer to our [paper](https://arxiv.org/pdf/2110.14890.pdf).

## Overview

Here is a README for the baselines in wikikgv2.

## Steps

1. Install the repo following [README.md](https://github.com/google-research/smore/blob/main/README.md).
2. Download and preprocess the dataset using `smore/cpp_sampler/download_and_preprocess_wikikgv2.py`.
3. Run TransE/ComplEx scripts under `smore/training/vec_scripts` and `smore/training/complex_scripts`. Baselines include `train_[shallow|mpnet|concat]_wikikgv2.sh`. Also set the wikikgv2 save directory in the scripts.

## License
SMORE is licensed under the Apache License, Version 2.0.

This is not an officially supported Google product.

Contact hyren@cs.stanford.edu and hadai@google.com for questions about the repo.
