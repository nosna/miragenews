

# MiRAGeNews: Multimodal Realistic AI-Generated News Detection

<p align="center">
  <a href="https://arxiv.org/abs/2410.09045"><img src="https://img.shields.io/badge/arXiv-2410.09045-b31b1b.svg"/></a>
  <a href="https://huggingface.co/datasets/anson-huang/mirage-news"><img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000"/></a>
  
  <br>
  <br>
  
  <img src="https://github.com/user-attachments/assets/1c934108-4c61-4495-8fd9-ee91d52bcec8" width="50%" />
</p>


## Abstract
The proliferation of inflammatory or misleading "fake" news content has become increasingly common in recent years. Simultaneously, it has become easier than ever to use AI tools to generate photorealistic images depicting any scene imaginable. Combining these two -- AI-generated fake news content -- is particularly potent and dangerous. To combat the spread of AI-generated fake news, we propose the MiRAGeNews Dataset, a dataset of 12,500 high-quality real and AI-generated image-caption pairs from state-of-the-art generators. We find that our dataset poses a significant challenge to humans (60% F-1) and state-of-the-art multi-modal LLMs (< 24% F-1). Using our dataset we train a multi-modal detector (MiRAGe) that improves by +5.1% F-1 over state-of-the-art baselines on image-caption pairs from out-of-domain image generators and news publishers. We release our code and data to aid future work on detecting AI-generated content.

## ***MiRAGeNews*** Dataset
***MiRAGeNews*** dataset contains a total of 15,000 pieces of real or AI-generated multimodal news (image-caption pairs) -- a training set of 10,000 pairs, a validation set of 2,500 pairs, and five test sets of 500 pairs each. Four of the test sets are out-of-domain data from unseen news publishers and image generators to evaluate detector's generalization ability.

<br>

Download ***MiRAGeNews*** from [HuggingFace](https://huggingface.co/datasets/anson-huang/mirage-news):
```py
from datasets import load_dataset

dataset = load_dataset("anson-huang/mirage-news")
```

## MiRAGe Detectors
We will release three detectors for different modalities: **MiRAGe-Img** for Image-only Detection, **MiRAGe-Txt** for Text-only Detection, and **MiRAGe** for Multimodal Detection. Pretrained models and code will be available soon.

Our detectors are more robust on out-of-domain (OOD) data from unseen news publishers and image generators than SOTA MLLMs and detectors.

<p align="center">
  <img src="https://github.com/user-attachments/assets/cac9691f-5ba5-4351-b1d9-a2ac679435b7" width="50%" />
  <img src="https://github.com/user-attachments/assets/778ef402-dbac-4b0c-a060-3fabaa6f15f3" width="49%" /> 
</p>

<img width="100%" alt="Screenshot 2024-10-15 at 5 57 53 AM" src="https://github.com/user-attachments/assets/eb4cea74-7210-4ae8-a58d-a2119dca3ab2">


## Acknowledgement
This research is supported in part by the Office of the Director of National Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA), via the HIATUS Program contract #2022-22072200005. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of ODNI, IARPA, or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for governmental purposes notwithstanding any copyright annotation therein.
