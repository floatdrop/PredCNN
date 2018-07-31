# PredCNN: Predictive Learning with Cascade Convolutions in Tensorflow 

PredCNN is a kind of combination of Temporal Convolutional Networks described in work [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271) by Shaojie Bai, J. Zico Kolter and Vladlen Koltun and [Video Pixel Networks](https://arxiv.org/abs/1610.00527) by
Nal Kalchbrenner, Aaron van den Oord, Karen Simonyan, Ivo Danihelka, Oriol Vinyals, Alex Graves, Koray Kavukcuoglu.

This repository contains a tensorflow implementation of the overlapping PredCNN architecture proposed in the [paper](https://www.ijcai.org/proceedings/2018/0408.pdf7).

Repository is still under heavy overhaul – some changes have been made to accomodate larger datasets (around 30Gb), some refactoring applyed.