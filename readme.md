## This is an implementation of Armijo Back-tracking line search on Training DNN
Implemented by Shigeng Sun, Sept 2022
At this moment this code is undergoing heavy development and testing. 

The implementation is heavily based on the original SGD implementation in Pytorch for the purpose of retaining the original code design and workflow for ease of deployment as an optimizer.

### This code contains a custom pytorch optimizer object SDLS (Stochastic Descent w. Line Search, SDLS.py)
### and a sample example code for training DNN on Fashin Mnist Data (minst_ls.py)

Besides producing a probable step acceptance mechanism for training DNNs, the focus is also to demonstrate one relatively simple way of incorperating line search into production of large machine learning solvers, so as to 


### *invite more researchers from ML and optimizaton alike to the theory and practice of optimization and machine learning.*

In short, the algorithm accepts a step if the armijo condition is satisfied:
$$f_k(x_k) - f_k(x_k+ \alpha g_k) > c_0\alpha \|g_k\|^2$$
Otherwise,
$$\alpha /= \tau$$
then the armijo condition is checked again. 
(here $f,g$ are the stochastic objective and gradient, respectively.)

The code incorperates options to use sample consistency/inconsistency, as well as other techniques to handel noisy stochastic observations.
The code skips an iterate if it thinks that the step would be too small to accept.


In this project, we investigate the effect of using Armijo back-tracking line search for training of large deep neural networks in PyTorch. This project encompass a custom implementation for an optimizer object which is compatible with training DNNs using standard SGD directions. While line search in optimizer object in PyTorch can be highly non-trivial, this project demonstrate a clean and relatively easy way of implementation of this task. It introduces the idea of sample consistency and inconsistency, incorporated line search relaxation ideas and produce reliable results. It can also be used as a tutorial to obtaining more information in the forward and backward pass at run time of the training process. 
