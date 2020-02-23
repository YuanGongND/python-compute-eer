# python-compute-eer
Simple Python script to compute equal error rate (EER) for machine learning model evaluation.

Usage:
```python
from compute_eer import compute_eer
label = [1, 1, 0, 0, 1]
prediction = [1, 1, 0, 1, 0]
eer = compute_eer(label, prediction)
print('The equal error rate is {:.3f}'.format(eer))
```
