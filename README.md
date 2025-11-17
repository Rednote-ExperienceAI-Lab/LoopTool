<h1 align="center"> <img src="./figures/logo_LoopTool.png" width="270" style="vertical-align:middle;"/><br>Closing the Data–Training Loop for Robust LLM Tool Calls</a></h1>

<div align="center"> 

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2511.09148v1)
[![Paper](https://img.shields.io/badge/Paper-Hugging%20Face-yellow?logo=huggingface)](https://huggingface.co/papers/2511.09148)
[![License](https://img.shields.io/badge/LICENSE-MIT-green.svg)](https://opensource.org/licenses/MIT) 
[![Python 3.10+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
</div>

<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

<!-- <div align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Orbitron&size=20&duration=3000&pause=1000&color=005DE3&center=true&vCenter=true&width=800&lines=Welcome+to+LoopTool;Closing the Data–Training Loop for Robust LLM Tool Calls;Powered+by+SJTU+Xiaohongshu+Inc." alt="Typing Animation" />
</div> -->

<div align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Orbitron&size=20&duration=3000&pause=1000&color=005DE3&center=true&vCenter=true&width=800&lines=Welcome+to+LoopTool;Closing+the+Data-Training+Loop+for+Robust+LLM+Tool+Calls;Powered+by+SJTU+Xiaohongshu+Inc.
  " alt="Typing Animation" />
</div>

## 📣 Latest News
- **[November 14, 2025]**: 🔥 LoopTool-8B is available on **[Hugging Face](https://huggingface.co/zhuiguang-ning/LoopTool-8B)**. LoopTool-32B, along with associated datasets, is scheduled to be released soon.
- **[November 13, 2025]**: 📄 Our paper is now available on **[arXiv](https://arxiv.org/abs/2511.09148v1)** and **[Hugging Face](https://huggingface.co/papers/2511.09148)**.
- **[November 13, 2025]**: 🚀 Our codebase released. You can use LoopTool to construct **specific dialogues for your own toolset**, andn finetune language model optimized for particular tools using RL algorithms. You can further refine performance by iteratively updating the training data and the model training process.

## 💡 Overview

Augmenting Large Language Models (LLMs) with external tools enables them to execute complex, multi-step tasks. However, tool learning is hampered by the static synthetic data pipelines where data generation and model training are executed as two separate, non-interactive processes. This approach fails to adaptively focus on a model's specific weaknesses and allows noisy labels to persist, degrading training efficiency. 

We introduce LoopTool, a fully automated, model-aware data evolution framework that closes this loop by tightly integrating data synthesis and model training. LoopTool iteratively refines both the data and the model through three synergistic modules: (1) Greedy Capability Probing (GCP) diagnoses the model's mastered and failed capabilities; (2) Judgement-Guided Label Verification (JGLV) uses an open-source judge model to find and correct annotation errors, progressively purifying the dataset; and (3) Error-Driven Data Expansion (EDDE) generates new, challenging samples based on identified failures. This closed-loop process operates within a cost-effective, open-source ecosystem, eliminating dependence on expensive closed-source APIs. 

Experiments show that our 8B model trained with LoopTool significantly surpasses its 32B data generator and achieves new state-of-the-art results on the BFCL-v3 and ACEBench benchmarks for its scale. Our work demonstrates that closed-loop, self-refining data pipelines can dramatically enhance the tool-use capabilities of LLMs.

### 📊 Overall Performance

<div align="center">
  <img src="./figures/overall_results_bfcl.png" width="96%" />
</div>
<br>

<div align="center">
  <img src="./figures/overall_results_acebench.png" width="96%" />
</div>
<br>

We compare LoopTool-8B and LoopTool-32B with various representation models in [BFCL-v3](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html) and [ACEBench](https://chenchen0103.github.io/ACEBench/). We adopt the official evaluation script and report the average accuracy across categories. On both BFCL-v3 and ACEBench leaderboards, LoopTool-8B achieves SOTA performance among all 8B-scale open-source models and exceeds several larger counterparts. LoopTool‑32B achieves the top position in BFCL‑v3, demonstrates the best performance among open‑source models in ACEBench, and ranks second overall, immediately following GPT‑4o.

### ✨ The LoopTool Framework

![Framework](<./figures/framework.png>)

**Key Features:**
- **Seed Data Generation**: We support a customizable toolset, utilizing a multi‑agent simulation framework to generate dialogue flows centered on a specific toolset. The overall architecture comprises the Planner, User, Assistant, and Tool.

- **End-to-End RL Training with [Verl](https://github.com/volcengine/verl)**: We support the transformation of specific tool‑invocation dialogues into samples adapted for model reinforcement learning training. By integrating the Verl library, we provide supervision over the model’s tool‑invocation steps.

- **Iterative Data and Model Evolution**: LoopTool iteratively refines both the data and the model through three synergistic modules: (1) Greedy Capability Probing (GCP) diagnoses the model's mastered and failed capabilities; (2) Judgement-Guided Label Verification (JGLV) uses an open-source judge model to find and correct annotation errors, progressively purifying the dataset; and (3) Error-Driven Data Expansion (EDDE) generates new, challenging samples based on identified failures.

## 🔧 Installation

###  Environment Setup
```bash
# Create conda environment
conda create -n looptool python=3.10
conda activate looptool

# Install requirements
cd LoopTool
pip install -r requirements.txt
```
We recommend following the official [Verl guidance](https://verl.readthedocs.io/en/latest/start/install.html) to correctly install the verl library.

## 📦 Seed Data Preparation
The scripts of seed data generation are organized in `dialog_generation` folder (`cd dialog_generation`). 

- The `tools` folder contains the complete set of tools. You may extendively incorporate custom tool sets. We provide 1.2w tool set descriptions scraped from [ToolBench](https://github.com/OpenBMB/ToolBench).
- Run the data generation:
```bash
    python run.py --func data_crawl --raw_data_path ./dialogs/toolbench --tools_path tools/toolbench --used_models Qwen3-32B --is_en --use_plan --use_cot --thread_num 10 --crawl_num 1000
```

- Transform the dialogs into conversations:
Please configure the file path properly.
```bash
    # Please configure the file path properly.
    python trans2conversation.py
```

## 🚀 Training
The scripts of GRPO training are organized in `grpotool` folder.

- Transform the conversation into GRPO training samples

```bash
    cd grpotool/dataset

    # Please configure the file path properly.
    python utils/conversation_transform_grpo_qwen.py 

    # （Option） filter out the training sample
    python utils/filter_grpo_sample.py

    # Please configure the file path properly.
    python utils/json2parquet.py
```
- GRPO Training for Robust Tool call
```bash
    # Qwen3-8B (8 H800)
    bash train_grpo_qwen.sh

    # Qwen3-32B (32 H800)
    bash multinode_qwen32b.sh
```

## 🔄 Data and Model Iteration
The scripts of data iteration are organized in `dataloop` folder. Please configure the file path properly according to your dataset.

- Greedy Capability Probing (GCP) queries the fine-tuned model on the training corpus using greedy decoding, revealing mastered, borderline, and failure cases. 

- Judgement-Guided Label Verification (JGLV) employs Qwen3-32B to compare each prediction against its reference label.

- Error-Driven Data Expansion (EDDE) transforms verified failure cases into new, structurally similar but contextually diverse challenging samples.

```bash
    cd dataloop

    # Greedy Capability Probing
    python greedy_capability_prob.py 

    # Judgement-Guided Label Verification (JGLV)
    python judgement_label_verification.py

    # Error-Driven Data Expansion (EDDE)
    python error_data_expansion.py

    # (Option) filter the new-generated samples
    python filter_grpo_sample.py
```

Upon obtaining a new round of data, we load the checkpoint from the previous round’s model and employ the new data to conduct GRPO reinforcement learning training.

## Evaluation

### BFCL
We recommend following the [official BFCL guidelines](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) to configure the evaluation environment, and employing the handler located in bfcl/qwentool.py as the Handler during our model evaluation.

### ACEBench
In the ACEBench evaluation, we instruct the model to produce tool calls of the `<tool_call> </tool_call>` type in order to align with the training format of the model.


## 🙏 Acknowledgement
We sincerely appreciate the contributions of the open-source community:
- [Verl](https://github.com/volcengine/verl)
- [ToolRL](https://github.com/qiancheng0/ToolRL)
- [ToolBench](https://github.com/OpenBMB/ToolBench)
- [BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
- [ACEBench](https://github.com/chenchen0103/ACEBench/)


## 📝 Citation

If you find this work helpful, please consider to cite our paper:
```bibtex
@misc{zhang2025looptool,
      title={LoopTool: Closing the Data-Training Loop for Robust LLM Tool Calls}, 
      author={Kangning Zhang and Wenxiang Jiao and Kounianhua Du and Yuan Lu and Weiwen Liu and Weinan Zhang and Lei Zhang and Yong Yu},
      year={2025},
      eprint={2511.09148},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.09148}, 
}
```

## 📄 License

This project is released under the [MIT License](LICENSE).

## 📞 Contact

For any questions or feedback, please reach out to us at [zhangkangning@sjtu.edu.cn](zhangkangning@ruc.edu.cn).
