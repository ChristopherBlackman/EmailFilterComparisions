### data set : https://www.kaggle.com/karthickveerakumar/spam-filter
### project repo : https://github.com/ChristopherBlackman/EmailFilterComparisions

### Author : Christopher Blackman 

### important files
- report : located in '''documents''' and is the only.pdf
- model 1-8 : are the initialization and training of the different models used in the report
- *.ipynb : a juypeter notebook used for figure generation
- modules/data_extractor : used for retrieving data, max-normalization, min-max-norm, and pca transform
- models/email_model : FFNN design used in the expirements

### requirements 
- python3, tensoflow, sklearn, numpy and any other basic libraries to build and run models
- juypeter notebook for building figures, and graphs

### Instructions 
- Download the dataset from listed Kaggle Website Above 
- run  	'''make nerual_network_selection''' for FFNN
- run 	'''make other_networks''' for naive Bayes, and Random Forest Models 
- after the above two commands have been completed, csv data will be avaible in 'output'
- to generate figures, run juypeter notebook, and rerun all the notebook files

