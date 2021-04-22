# A Hierarchical Multi-output Deep Neural Network for Image Classification

Author: Edwin S Yan

_Individual Research Project - EN.605.746 - Advanced Machine Learning, Johns Hopkins University_


## Environment
There are two ways to reproduce my environment in Windows:
1. **conda_env.yml** - Conda Environment Export
2. **requirements.txt** - Pip Packages Export


## Folder Structure
 - CIFAR 10:
    - Flat Approach
    - Hierarchical Approach
 - CIFAR 100:
    - Flat Approach
    - Hierarchical Approach
 - CAR 196:
    - Flat Approach
    - Hierarchical Approach

## Abstract
A hierarchical data structure is ubiquitous in many real-world applications to organizeand memorize information.  Hierarchical classification is a particular type of classificationproblem to predict multiple class labels within a hierarchical tree.  It has comprehensive usecases in image classification tasks.  Simultaneously, the hierarchical classification problemis also very challenging to ensure hierarchical consistency among different labels.  Histor-ically, these hierarchical structures have widely been ignored.  The model will either onlypredict the most granular level then recover the parent classes, or simply predict each classindividually.   Neither  approach  is  ideal  due  to  the  trade-off  between  accuracy  and  hier-archical consistency.  This paper proposes a novel Hierarchical Multi-output Deep NeuralNetwork (HM-DNN) architecture to uncover hierarchical structure by concatenating em-bedding space between parent class node and child class node.  The hierarchical structureis discovered in the fully connected layers so that HM-DNN can be used with any popu-lar Deep Neural Network architectures for image classifications, such as VGG (Simonyanand Zisserman, 2014) and ResNet (He et al., 2015).  I implement HM-DNN on CIFAR-10(Krizhevsky, 2012), CIFAR-100, and CAR-196 (Krause et al., 2013) datasets to comparewith the traditional implementation of the multi-output DNN. The experimental resultsindicate  that  HM-DNN  has  a  significant  improvement  in  hierarchical  consistency  and  aslight improvement in overall accuracy compared to the baseline approach.