# MachineLearning-Homework-NeuralNetwork

## Dependencies
- Python3
- Tensorflow 
- TFLearn

## Goal of the classification 
1) Identification of public services boats (Ambulance, Police, Fireman,...) aginst private boats. 

## Results until now
### boat-classification ver 2
- Code run with 25 epoch: **95% accuracy | 32% precision | 63% recall** - [35 true positive , 75 false positive, 21 false negative]

- Code run with 20 epoch and 0.2 dropout probability (at both layers): **97% accuracy | 60% precision | 11% recall** - [6 true positive, 4 false positive, 50 false negative]  /  same code with 25 epoch: **96,3% accuracy | 40,7% precision | 62,5% recall** - [35 true positive, 51 false positive, 21 false negative]

- Code run with 20 epoch, only one fully connected layer and 0.2 dropout: **96.3% accuracy | 22.5% precision | 12.5% recall** - [7 true positive, 24 false positive, 49 false negative]

- Code run with optimizer '*Adam*' or '*Stochastic Gradient Descendt*' instead of '**Momentum**' and 25 epoch -- totail failure, no positive predictions (all 0)
