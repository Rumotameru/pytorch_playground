# Pytorch_playground
### Select the network parameters and get a clear result of its classification.

    python 3.7.4
    pandas 0.25.1
    sklearn 0.21.3
    torch 1.8.1+cu111	
    matplotlib 3.1.1	

Main body of the program is in Main.py. The rest of the modules contain only the functions used. The iris dataset of the sklearn library was used as the data for this project. 

There are 6 selectable parameters:
  * __"additional_features"__ (list): the list of additional features that will be used in the learning process in addition to "x1" (1st component of PCA) and "x2" (2nd component of PCA) that are already selected. *"x1", "x2", "x1^2", "x2^2", "x1x2", "sin(x1)","sin(x2)" are available.*
  * __"layers"__ (list): the list of ouput neurons' numbers for each layer.
  * __"activator"__(str): activation function that will be used for all layers of your "toy" model. *"relu", "tanh", "sigmoid", "linear" are available. "linear" is default.*
 "linear" corresponds to the variant of the model without an activator
  * __"epochs"__ (int): number of epochs for train-valid process.
  * __"learning_rate"__ (float): step size at each iteration of optimization.
  * __"batch_size"__ (int): number of training examples utilized in one iteration.

Сhange the required parameters in the dictionary and run the program. As the result you'll get accuracy score of the model and plot where the original classes are indicated by transparent markers, and the predicted ones are displayed on top of them.![image](https://user-images.githubusercontent.com/22664697/121881628-35c5d980-cd18-11eb-93e5-6cbb4b263a7a.png)
