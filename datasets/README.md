# Datasets

This directory contains partial datasets used or adapted for our project.

## Included Datasets

### minif2f  
**Original Source**: [miniF2F](https://github.com/deepseek-ai/DeepSeek-Prover-V1.5/blob/main/datasets/minif2f.jsonl)  
**License**: MIT License  

We have adapted parts of the dataset from DeepSeek-Prover-V1.5 to fit our task format and experimental pipeline.  
Modifications include:
- Some incorrect questions have been modified, which is mentioned in the paper.
- Modified the header

Please refer to the original repository and license file for full details:
- [Original Repository](https://github.com/deepseek-ai/DeepSeek-Prover-V1.5)
- [LICENSE](https://github.com/deepseek-ai/DeepSeek-Prover-V1.5/blob/main/LICENSE)

## Usage

Datasets are stored in JSONL format. You can load them using:

```python
from dsp.utils import load_dataset

datasets = load_dataset("path/to/your/dataset", "your_split")
```