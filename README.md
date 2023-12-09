<div align="center">

## <span style="color:blue; font-size:48px;">💻데이터 기반의 머신러닝 기법을 이용한 Optimization⌛</span>

</div>


# Introduction
* 다양한 분야에서 최적화를 할 때에 사용자 또는 실험자가 일일히 데이터를 얻어서 그에 따른 경험 및 지식에 의존에서 목표로 하는 성능이 나올때까지 이 과정을 반복하게 된다. 
* 따라서 이 과정은 많은 시간과 비용이 소모되는데, 여기서 데이터 기반의 머신러닝을 이용한 최적화 방법으로 시간과 비용을 줄이고자 한다.
* 여기선 위스콘신 유방암 진단데이터를 가지고 종양의 면적을 최대화 시키는 조건을 베이지안 최적화 방법으로 알아보고 어떤 요소가 가장 영향을 크게 미치는지 분석해보고자 한다.

* # Contents

* [Bayesian Optimization](#Bayesian-Optimization)
* [Surrogate Model](#Surrogate-Model)
* [Acquisition Function](#Acquisition-Function)
* [Installation](#Installation)
* [Usage](#Usage)
* [Dataset](#Dataset)
* [License](#License)
* [References](#References)
* [Future Work](#Future-Work)
---
# Bayesian Optimization

* **"Bayesian Optimization (베이지안 최적화)"** 는 어떠한 수학적 표현식을 알 수 없는 목적 함수 f **(black box function)** 의 최댓값 또는 최솟값을 최적화하는 방법 중 하나이다.
* 베이지안 최적화는 샘플링된 학습 데이터를 사용하여 미지의 목적함수에 대한 **surrogate model**을 구축하고, 목적함수의 불확실성을 최소화하거나 함수의 예측값을 최대화할 수 있을 것으로 기대되는 다음 포인트를 **acquisition function**으로 선정하여 그 포인트에서 데이터를 다시 샘플링하여 **surrogate model**을 업데이트하는 반복과정으로 목적함수의 최댓값 또는 최솟값일 것으로 예상되는 조건(parameter)을 제시해준다. 그리고 user는 이 모델이 제시해주는 조건을 사용해 목적함수 값을 얻을 수 있다.

<table>
  <tr>
    <td>
      <img src="https://github.com/Jaewan-Baek/Optimization/assets/144581812/0a859fd4-77d2-472c-bad6-7ec737f57e9e.jpg" width="500" height="300" title="Black box function">
      <br>
      <span style="font-size: 10px;">Image by Julie Bang © Investopedia</span>
    </td>
    <td>
      <img src="https://github.com/Jaewan-Baek/Optimization/assets/144581812/fe510135-7aad-475f-a6cd-26db6088fbe1.jpg" width="500" height="300" title="Black box function">
      <br>
      <span style="font-size: 10px;">https://gils-lab.tistory.com/61</span>
    </td>
  </tr>
</table>

* 베이지안 최적화의 motivation은 머신러닝의 하이퍼파라미터 튜닝에서 발전해왔으며, 현재는 다양한 분야(촉매, 설계, 전지 등)에서도 이 방법을 적용시키고자하는 연구가 진행되고 있다.
* Local maximum/minimum 보다는 **Global maximum/minimum** 최적화에 적합하며, 일반적으로 파라미터의 수가 많을수록 더욱 장점이 드러난다.
  
<p align="center">
  <img src="https://github.com/Jaewan-Baek/Optimization/assets/144581812/6b55af5e-4a27-4a33-a9fd-fa996066b12e" width="700" height="500" title="Black box function">
  <br>
  <span style="font-size:3px;">[Source](https://www.kaggle.com/code/clair14/tutorial-bayesian-optimization#Test-it-on-data)</span>

# Surrogate Model
* surrogate model 은 샘플링된 데이터를 획득하여 목적 함수를 추정하는 머신러닝 모델이며, 주로 **Gaussian process**를 활용한다.
* **Gaussian process**는 커널(kernel)을 이용하고, 평균과 공분산이라는 개념을 이용해 불확실성 영역을 형성한다. 데이터를 획득할때마다 평균과 공분산이 달라져 모델이 업데이트 된다.

<p align="center">
  <img src="https://github.com/Jaewan-Baek/Optimization/assets/144581812/192014f4-bedf-4b40-acfc-1285ad8cddf2" width="500" height="300" title="Black box function">
  <br>
  <span style="font-size:3px;">[Source](https://arxiv.org/pdf/1012.2599.pdf)</span>

# Acquisition Function
* surrogate model 을 통한 목적함수에 대한 현재까지의 확률적 추정 결과를 바탕으로 다음 파라미터의 집합을 사용자에게 제시해준다.
* **탐색(exploration)** 과 **활용(exploitation)** 이 두 성질을 조절해 이용하며, 이에 따라 여러 기법이 있는데 대표적으로 **EI(Expected Improvement)**, **PI(Probability of Improvement)**, **LCB(Lower Confidence Bound)** 등이 있다.
 <p align="center">
  <img src="https://github.com/Jaewan-Baek/Optimization/assets/144581812/bb3b9fea-b822-4960-9bad-153abd6b6aa8" width="500" height="400" title="Bayesian Optimization">
  <br>
  <span style="font-size:15px;">Image from https://www.iro.umontreal.ca/~bengjoy/cifar/NCAP2014-summerschool/slides/Ryan_adams_140814_bayesopt_ncap.pdf</span>

 <p align="center">
  <img src="https://github.com/Jaewan-Baek/Optimization/assets/144581812/2776ab59-98ff-4a57-9bed-900769784536" width="700" height="780" title="Bayesian Optimization">
  <br>
  <span style="font-size:15px;">Wikipedia.org </span>


# Installation
scikit-optimize requires

  * Python >= 3.6
  * NumPy (>= 1.13.3)
  * SciPy (>= 0.19.1)
  * joblib (>= 0.11)
  * scikit-learn >= 0.20
  * matplotlib >= 2.0.0

``` python
pip install scikit-optimize
```

# Usage
~~~ python
import numpy as np
from skopt import gp_minimize

def f(x):
    return (np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2)) +
            np.random.randn() * 0.1)

res = gp_minimize(f, [(-2.0, 2.0)])
print("x*=%.2f f(x*)=%.2f" % (res.x[0], res.fun))
~~~

for 루프를 적용할 수도 있다.
~~~ python
from skopt import Optimizer
opt = Optimizer([(-2.0, 2.0)])
for i in range(20):
    suggested = opt.ask()
    y = f(suggested)
    res = opt.tell(suggested, y)
print("x*=%.2f f(x*)=%.2f" % (res.x[0], res.fun))
x*=0.27 f(x*)=-0.15
~~~

# Dataset
* 사용한 데이터셋은 위스콘신 유방암 진단데이터이다.
* 원본 데이터에서 30개의 특성 중 6개의 특성(figure)값과 596개의 데이터를 사용했다.
* https://github.com/Yorko/mlcourse.ai/tree/main/data


# License
This repository is licensed under the BSD 3-Clause license. See LICENSE for details.

Click here to see the License information --> [License](LICENSE)

# References
* [https://scikit-optimize.github.io/stable](https://scikit-optimize.github.io/stable/)
* Gan, Weiao, Ziyuan Ji, and Yongqing Liang. "Acquisition functions in bayesian optimization." 2021 2nd International Conference on Big Data & Artificial Intelligence & Software Engineering (ICBASE). IEEE, 2021.
* Snoek, Jasper, Hugo Larochelle, and Ryan P. Adams. "Practical bayesian optimization of machine learning algorithms." Advances in neural information processing systems 25 (2012).
* Brochu, Eric, Vlad M. Cora, and Nando De Freitas. "A tutorial on Bayesian optimization of expensive cost functions, with application to active user modeling and hierarchical reinforcement learning." arXiv preprint arXiv:1012.2599 (2010).


# Future Work
* 추후 연구로는 Chemical Synthesis 분야에 기계 학습을 적용시켜 화학 구조 최적화를 해볼 예정입니다!😊


