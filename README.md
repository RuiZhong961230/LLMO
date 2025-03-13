# LLMO
Large Language Model Assisted Adversarial Robustness Neural Architecture Search

## Abstract  
Large Language Models (LLMs) have shown significant promise as evolutionary optimizers. This paper introduces a novel LLM-based Optimizer (LLMO) to address Neural Architecture Search Considering Adversarial Robustness (ARNAS), a classic combinatorial optimization problem. Using the standard CRISPE framework, we design the prompt and employ Gemini to iteratively refine solutions based on its responses. In our numerical experiments, we investigate the performance of LLMO on NAS-Bench-201-based ARNAS tasks with CIFAR-10 and CIFAR-100 datasets. The results, compared with six well-known metaheuristic algorithms (MHAs), highlight the superiority and competitiveness of using LLMs as combinatorial optimizers. The source code is available at https://github.com/RuiZhong961230/LLMO.

## Citation
@INPROCEEDINGS{Zhong:24,  
  author={Rui Zhong and Yang Cao and Jun Yu and Masaharu Munetomo},  
  booktitle={2024 6th International Conference on Data-driven Optimization of Complex Systems (DOCS)},  
  title={Large Language Model Assisted Adversarial Robustness Neural Architecture Search},  
  year={2024},  
  volume={},  
  number={},  
  pages={433-437},  
  doi={10.1109/DOCS63458.2024.10704419}  
  }

## Datasets and Libraries
The gemini-pro API is provided by the google.generativeai==0.0.1 library, and the dataset of adversarial robustness neural architecture search (ARNAS) is downloaded from https://steffen-jung.github.io/robustness/

## Contact
If you have any questions, please don't hesitate to contact zhongrui[at]iic.hokudai.ac.jp
