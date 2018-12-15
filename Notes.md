# Notes
## Brief history of what we finally got right

- The final valid_loss is not strictly related to network's performances. One test with final valid_loss > 0.5 had same accuracy of another one with final valid_loss = 0.09. 

- Number of epoch is related to dataset dimension, increasing the number of epoch with small dataset induce overfitting. 
