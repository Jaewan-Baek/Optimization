# Bayesian Optimization

* **"Bayesian Optimization (베이지안 최적화)"** 는 어떠한 수학적 표현식을 알 수 없는 목적 함수 f(black box function) 의 최댓값 또는 최솟값을 최적화하는 방법 중 하나이다.
* 베이지안 최적화는 샘플링된 학습 데이터를 사용하여 미지의 목적함수에 대한 **surrogate model**을 구축하고, 목적함수의 불확실성을 최소화하거나 함수의 예측값을 최대화할 수 있을 것으로 기대되는 다음 포인트를 **acquisition function**으로 선정하여 그 포인트에서 데이터를 다시 샘플링하여 surrogate model을 업데이트하는 반복과정으로 목적함수의 최댓값 또는 최솟값일 것으로 예상되는 조건(parameter)을 제시해준다.

 
<p align="center">
  <img src="https://github.com/Jaewan-Baek/Optimization/assets/144581812/0a859fd4-77d2-472c-bad6-7ec737f57e9e" width="500" height="300" title="Black box function">
  <br>
  <span style="font-size:3px;">Image by Julie Bang © Investopedia, 2019</span>

 <p align="center">
  <img src="https://github.com/Jaewan-Baek/Optimization/assets/144581812/2776ab59-98ff-4a57-9bed-900769784536" width="700" height="780" title="Bayesian Optimization">
  <br>
  <span style="font-size:15px;">Wikipedia.org </span>

---

# Concept



# Contents
* [Installation](#Installation)
* [Surrogate Model](#Surrogate-Model)
* [Acquisition Function](#Acquisition-Function)
* [Usage](#Usage)
* [Dataset](#Dataset)
* [License](#License)
* [References](#References)
* [Future Work](#Future-Work)
