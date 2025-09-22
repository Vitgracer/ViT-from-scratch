![Visits](https://api.visitorbadge.io/api/VisitorHit?user=Vitgracer&repo=vit-from-scratch&countColor=%237B1E7A)
![GitHub last commit](https://img.shields.io/github/last-commit/Vitgracer/vit-from-scratch?color=blue)
![GitHub repo size](https://img.shields.io/github/repo-size/Vitgracer/vit-from-scratch?color=green)
![GitHub stars](https://img.shields.io/github/stars/Vitgracer/vit-from-scratch?style=social)
![GitHub forks](https://img.shields.io/github/forks/Vitgracer/vit-from-scratch?style=social)
![Python](https://img.shields.io/badge/Python-3776AB.svg?logo=python&logoColor=white)

# Simple minimal Visual Transformer implementation in PyTorch
Hey Friends! 

Welcome to this tiny experiment where we compare a **classic Convolutional Neural Network (CNN)** against the modern **Vision Transformer (ViT)**.
The task: old and gold handwritten digits recognition.

Why digits? ~~Because my PC will explode with anything more serious~~ 😂 Because the goal here is not to ACHIEVE, but to **see how ViTs actually work under the hood**. 

---

## ⚙️ Installation  

Create a virtual environment and install dependencies:  

```bash
python -m venv vit-venv
source vit-venv/Scripts/activate
pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip3 install einops torchsummary
```

How tu run: 
```bash
python train.py
```

## 🥊 CNN vs ViT

We train two models with roughly **~2k** trainable parameters each.

🔵 CNN! the test accuracy looks like this:

🔴 ViT! And here’s the ViT’s performance:

## 🤔 Observations
- Attention layers involve matrix multiplications of full sequences (O(N²) complexity), so **ViT is SLOWER**. Not like a turtle.. but the turtle loaded with bags from supermarket 😂
- The **ViT also gets lower accuracy**, because Transformers work good when they can model long-range dependencies and are fed with lots of data. On MNIST, the images are tiny (28×28) and the dataset is small. CNNs are simply better at extracting local patterns like edges, strokes, and curves.

**In other words**: asking a ViT to classify MNIST is like hiring a theoretical physicist to count apples at a grocery store. So choose your model wisely! 🧐

## ⚖️ Pros & Cons
**CNN** 
- ✅ Fast to train
- ✅ Great at local pattern recognition (edges, textures, shapes)
- ✅ Works very well with small datasets
- ❌ Limited ability to capture global context

**ViT**

- ✅ Elegant, unified architecture (no handcrafted kernels)
- ✅ Scales with data (huge datasets)
- ✅ Attention maps are interpretable (attention weights show where the model is “looking”)
- ❌ Training is slower
- ❌ Needs more data to reach CNN-level performance

<!-- python -m venv vit-venv
source vit-venv/Scripts/activate 

pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip3 install einops, torchsummary

Say that we are soing simple digits prediction, because the main goal for us to show how ViT works.. 

say that we have a simple cnn and its accuracy for test dataset is 98.95%.

Say taht boty model have ~2k params 

Say why transformers are better
Or why to chosse CNNs
Make a copmparison  -->
