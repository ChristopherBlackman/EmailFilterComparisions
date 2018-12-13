from modules.email_model import FFNN,main
from modules.data_extractor import extractor 
import pandas as pd

def main(name,args={},norm="minmax",dim=1500): 
    print("Importing Data")
    if norm == "minmax":
        ((trX, trY),(teX, teY)) = extractor(file='data/emails.csv').min_max_nomalized()
    elif norm == "pca":
        ((trX, trY),(teX, teY)) = extractor(file='data/emails.csv').pca_reduced_nomarlize(dim=dim)
    print("Creating Model")
    model = FFNN(**args)
    print("Training Model")
    print(type(trX))
    acc , t, lr= model.train(trX,trY,teX,teY)

    df = pd.DataFrame({ 'accuracy':acc,
                        'epoc':list(range(len(acc))),
                        'time_per_iteration(s)':t,
                        'learning_rate':lr})

    df.to_csv("output/{0}.csv".format(name))

args = {
        'initial_learning_rate':0.5, 
        'hidden_layer_size':60, 
        'input_size':33485, 
        'output_size':2, 
        'drop_percent':0.5, 
        'drop_after_epochs':3,
        'epochs':50
        }

main(__file__.split(".")[0],args=args,norm="minmax",dim=1500)
