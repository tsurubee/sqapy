# sqapy
Simulated quantum annealing implemented in Python

## Installation
```
pip install sqapy
```

## Usage
### Example
```python
import numpy as np
import sqapy

b = np.array([10,10,-10,-10])
c = np.array([10,-10])
W = np.array([[5,-5],[5,-5],[-5,5],[-5,5]])

model = sqapy.BipartiteGraph(b, c, W)
sampler = sqapy.SQASampler(model, steps=100)
energies, spins = sampler.sample(n_sample=3)
```

## License
This project is licensed under the terms of the MIT license, see LICENSE.