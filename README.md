# Optimizing Deep Learning Models: From CNN Quantization to Smol LLM

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Project Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)](https://github.com/Faisal-repo/Empty)

**A comprehensive initiative combining model quantization, few/zero-shot learning, and knowledge distillation to build efficient deep learning models**

---

## 🚀 Current Progress (Updated: 06/02/2025)

```mermaid
flowchart LR
    subgraph Step1
        A["✔️Project Start"] --> B("✔️ Environment Setup")
    end

    subgraph Step2
        direction TB
        B --> C["✔️ Baseline CNN"]
        C --> D(["✔️ FP16 Training"])
    end

    subgraph Step3
        direction LR
        E --> F[("✔️ Hardware-Aware<br>Quantization")]
    end

    subgraph Step4
        direction TB
        F --> G{"🔄 Knowledge<br>Distillation"}
        G --> H{{"🔄 Few/Zero-Shot<br>Learning"}}
    end

    Step1 --> Step2
    Step2 --> E[["✔️ QAT Implementation"]]
    Step3 --> Step4
    Step4 --> I[/"❌ Optimized Smol LLM"/]
```

**Legend:**  
✔️ Completed | 🔄 In Progress | ❌ Pending

---

## 🧪 Experiment Notebooks

### QAT on MNIST
- **[Initial Implementation](QAT%20test%20NodeBooks/quantization-aware-training-qat-on-mnist-error.ipynb)**  
  Early QAT implementation with error analysis and debugging

- **[Stable Version](QAT%20test%20NodeBooks/quantization-aware-training-qat-on-mnist-v2.ipynb)**  
  Final working QAT implementation with proper quantization handling

---

## 📌 Key Improvements Implemented

1. **Quantization Stability**
   - Fixed tensor layout issues using `.reshape()`
   - Implemented proper QConfig handling
   - Added layer fusion validation

2. **Reproducibility**
   - Pinned library versions
   - Added seed initialization
   - Created standardized evaluation scripts

3. **Documentation**
   - Added detailed roadmap documents
   - Maintained error analysis notebooks
   - Structured repository for clarity

---

## 🎯 Next Steps

1. **Knowledge Distillation Pipeline**
   - [ ] Implement teacher-student framework
   - [ ] Design distillation loss function
   - [ ] Optimize temperature parameter

2. **Smol LLM Integration**
   - [ ] Adapt quantization techniques for transformers
   - [ ] Implement dynamic quantization
   - [ ] Benchmark on text classification

3. **Repository Improvements**
   - [ ] Add experiment tracking with Weights & Biases
   - [ ] Create standardized training scripts
   - [ ] Add CI/CD pipeline for testing

---

## 📚 Documentation & Resources

- [Project Roadmap](🚀%20Major%20Project%20Roadmap%20CNN%20Quantization%20&%20Smol%20LLM%20Integration.md)
- [QAT Implementation Guide](QAT%20test%20NodeBooks/quantization-aware-training-qat-on-mnist-v2.ipynb)
- [Base Papers](Base%20Paper/)

---


## 📚 Metrices

| Model  | Accuracy | Fine-tuned | Top-5 Accuracy | Top-1 Accuracy | Link |
|--------|----------|------------|---------------|---------------|------|
| Resnet-18 | 81      | no        |           |         | [Resnet-18 Link](https://github.com/Faisal-repo/Empty/blob/main/QAT%20test%20NodeBooks/resnet-18_pytorch-lighting_with_fine-tune.ipynb) |
| Resnet-18(QAT) |       | No         |           |           | [Model B Link](https://example.com/modelB) |
| Model C | 88%      | Yes        | 96%           | 88%           | [Model C Link](https://example.com/modelC) |
| Model D | 80%      | No         | 90%           | 80%           | [Model D Link](https://example.com/modelD) |



## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

> **Note:** This README is dynamically updated as the project progresses. Last updated: 06/02/2025
