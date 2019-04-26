# MIL Noise Filter
Filter algorithm for noise reduction in multiple-instance classification problems

In contrasts to regular classification problems, in which each example has a unique description, in multiple-instance classification problems each example has many descriptions. Noisy data deteriorate classifiers performance in multiple-instance classification as it does in regular classification problems. Multiple-instance data may have noise at two different levels: noisy instances inside the bag and noisy bags in the training set. Here we implemented an algorithm that filters noisy instances inside bags. 

First, all instances are extracted from the training bags. Each instance is assigned to its bag's class label to create a regular training data set. A regular noise filter is applied to the new training data and noisy instances are removed from the set. Finally, the training data is mapped back to its multiple-instance representation. More details can be found in:

- Luengo, J., Sanchez Tarrago, D., Prati, R.C., Herrera, F.: A First Study on the Use of Noise Filtering to Clean the Bags in Multi-Instance Classification. In: Proceedings of the International Conference on Learning and Optimization Algorithms: Theory and Applications. pp. 3:1–3:6. ACM, New York, NY, USA (2018). <a href="https://dl.acm.org/citation.cfm?id=3230911&dl=ACM&coll=DL" target="_blank">(link)</a>

Developed with:
- Java 1.8
- NetBeans IDE 8.2

Dependencies:
- Weka 3.7
- Weka package citationKNN 1.0.1
- Weka package multiInstanceLearning 1.0.10
- Weka package multiInstanceFilters 1.0.10
