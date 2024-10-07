Small description of the main branch

## PyTorch

This directory contains various deep learning models implemented using PyTorch. For each model, you'll find a link to the repository that served as the basis for the implementation. The models include different types such as RNN, LSTM, GNN, CNN, and KAN. Each model accepts the corresponding batch data (either tabular or image data, depending on the model) and an edge input, which can be set to None for models that don't use graph data.

## Sklearn

This directory contains the Model wrapper class and its subclasses, which follow the Scikit-learn API and can be used with any model adhering to that interface. You can implement different types of model optimization, including GridSearchCV and BayesSearchCV. There are some specialized classes as well: ModelVoting and ModelStacked. The former builds an ensemble of models that predict the same target and returns the average of the predictions. The ModelStacked class stacks multiple models, with each model using the predictions from the previous model as input. Note that the ModelStacked class has not yet been tested.

## Src

## Test

Feel free to implement new models, new strategies or to use the models !
