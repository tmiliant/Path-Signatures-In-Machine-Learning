# Path signatures in machine learning, especially used in time series classification and prediction. My repository also appears for reference at https://paperswithcode.com/paper/a-primer-on-the-signature-method-in-machine and https://arxiv.org/abs/1603.03788. My research output is also made available in this repository. Please see the paper at https://ieeexplore.ieee.org/abstract/document/9747285. 

## Path signature

I use a pure mathematics tool called the "path signature" to build features for machine learning. 
There are plenty of resources for understanding this topic, here is the best one I know so far: https://arxiv.org/abs/1603.03788. Additionally, I am using the pysegments package https://github.com/datasig-ac-uk/pysegments, which is used to adapt the model's predictions to the detection of appliance usage.

Essentially, one can summarize a path using a vector of numbers. This vector is infinite, and under some easy-to-satisfy conditions, it uniquely determines the path. As computers cannot store infinity, we can truncate this vector empirically to decide how much of it we should keep in our data science use-cases. 

Then the truncated vector is the feature set of the path, and we can immediately turn this into a classification or a prediction problem. 

In this research project I used random forests and convolutional neural networks on these signatures.

## Project description - supervised by Dr Paul Moore and Dr Yue Wu

An electricity bill is usually displayed with the total consumption rather than with how much each individual appliance - fridge, boiler, microwave etc - consumes.

Therefore, it would be very useful to derive, by just using the voltage and current of the TOTAL consumption, the INDIVIDUAL appliance consumption.

As an example, the problem is: given the total consumption graph - voltage, current for all appliances - tell me how much the fridge consumed. It seems hard to derive how much the fridge consumed by considering all the consumptions of the fridge, microwave, boiler and other 20 appliances on top of each other.

I used the UK Dale dataset, publicly available. The preprocessing of this data turned out to be very challenging, because of the use of FLAC files, having one per hour. Thus each such file must be processed first, then all of them concatenated together to form a very large dataset. 

Train: each individual appliance with how much it consumed for a period of a few years.
Test:  given total consumption, derive how much did the fridge consume.

I obtained 94% accuracy on this. Accuracy is measured like this: proportion of period of time where I predict the `on`/`off` states of the fridge correctly. `on` just means that the monitored power of the fridge is >20W.

This shows the power of the path signature and its potential for other uses in machine learning.  

## Code

The data processing is very important, because it calculates the signatures. This can be done using the following packages:
https://github.com/bottler/phd-docs/blob/master/iisignature.pdf
https://pypi.org/project/esig/

I chose to use iisignature for this project, and noticed that the signatures were computed within miliseconds for a 3x320 path.

