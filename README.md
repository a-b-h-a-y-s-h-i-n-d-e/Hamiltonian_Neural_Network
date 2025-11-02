# Minimal Implementation of Hamilton Neural Network

A lightweight and educational implementation of **Hamiltonian Neural Networks (HNNs)** — a physics-informed neural model that learns the underlying *Hamiltonian function* from data, ensuring **energy conservation** and **physically consistent dynamics**.



### 🚀Usage Example

```py
import torch
from hnn import HNN
from trainer import Trainer
from mass_spring.data import get_dataset

data = get_dataset(test_split=0.8)

model = HNN()
trainer = Trainer(model)


trainer.fit(data['x'], data['dx'], epochs=100) 
```

### 🧩 Visualize Results

```py
import matplotlib.pyplot as plt
losses = trainer.losses
plt.title('HNN Training Loss over Epochs', fontsize=16, fontweight='bold')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.plot(losses)
```

<p align="center"> <img src="mass_spring/loss_curve.png" alt="HNN Training Loss Curve" width="600"> </p>

### 🧾 References

* Greydanus, S., Dzamba, M., & Yosinski, J. (2019).
Hamiltonian Neural Networks.
NeurIPS 2019. [https://arxiv.org/abs/1906.01563](https://arxiv.org/abs/1906.01563)

* Scibits Blog (JAX Implementation) - [article link](https://scibits.blog/posts/hnn/)

