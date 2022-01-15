# F-Principle

This is an exercise problem of the digital signal processing (DSP) course at
[School of Artificial Intelligence at the Nanjing University (NJU)](https://ai.nju.edu.cn/),
teaching by [Han-Jia Ye](http://www.lamda.nju.edu.cn/yehj/).
The course homepage is at [DSP](https://www.lamda.nju.edu.cn/yehj/dsp2021/).
This exercise is written by [Jia-Qi Yang](https://lamda.thyrixyang.com).
Please feel free to contact me
by mailing yangjq@lamda.nju.edu.cn if you have any questions.


## Problem 1: Understanding F-Principle (35pt)

Read following articles:

1. [F-Principle](https://ins.sjtu.edu.cn/people/xuzhiqin/fprinciple/index.html) 
2. [Frequency Principle: Fourier Analysis Sheds Light on Deep Neural Networks](https://ins.sjtu.edu.cn/people/xuzhiqin/pub/shedlightCiCP.pdf)

Then, answer following questions:

1. What is F-Principle ? (5pt)
2. Why F-Principle is important ? (5pt)
3. What are the differences between *response frequency* and *input frequency* ? Which one is used in F-Principle ? (5pt)
4. How is frequency defined in high-dimensional functions ? Why ? (10pt)
5. How does the authors verify F-Principle experimentally ? (10pt)

## Problem 2: Reproducing F-Principle (65pt)

Code to reproduce F-Principle by the authors is published at [F-Principle Github](https://github.com/xuzhiqin1990/F-Principle). 

You may modify [F-Principle Github](https://github.com/xuzhiqin1990/F-Principle) to conduct following experiments. 
However, this implementation is based on tf1.x, and the high-dim experiments are not implemented.
You may also choose to extend pytorch training scripts provided in src/.

### 2.1 Low-dim Experiment (25pt)

Read [F-Principle in low-dim experiments](https://ins.sjtu.edu.cn/people/xuzhiqin/fprinciple/ldexperiment.html).

1. Plot training procedure in Spatial Domain, i.e. the first figure in [F-Principle in low-dim experiments](https://ins.sjtu.edu.cn/people/xuzhiqin/fprinciple/ldexperiment.html). (10pt)
2. Plot training procedure in Fourier Domain, i.e. the second figure in [F-Principle in low-dim experiments](https://ins.sjtu.edu.cn/people/xuzhiqin/fprinciple/ldexperiment.html). (10pt)

You may plot several figures instead of gifs in [F-Principle in low-dim experiments](https://ins.sjtu.edu.cn/people/xuzhiqin/fprinciple/ldexperiment.html).

### 2.2 High-dim Experiment (30pt)

Read [F-Principle in high-dim experiments](https://ins.sjtu.edu.cn/people/xuzhiqin/fprinciple/hdexperiment.html).

1. Implement the *projection method* **or** the *filtering method* on MNIST dataset.
2. Describe the procedure of your method using pseudo-code.
3. Inspect how each response frequency component (e.g. high-frequency and low-frequency) converges.
You may plot figures or using tables to demostrate your results.

### 2.3 Summay (10pt)

1. What did you learn from this practice problem ? (5pt)
2. What problems did you encounter and how did you solve them ? (5pt)