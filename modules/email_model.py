import numpy as np
import tensorflow as tf
from modules.data_extractor import extractor 
from sklearn.model_selection import KFold
from datetime import datetime
import math
import time

class FFNN():
    def __init__(self,**kwargs):
        self.initial_learning_rate = kwargs.get('learning_rate',0.5) 
        self.hidden_layer_size = kwargs.get('hidden_layer_size',30) 
        self.input_size = kwargs.get('input_size',33485) 
        self.output_size = kwargs.get('output_size',2) 
        self.drop_percent = kwargs.get('drop_percent',0.5) 
        self.drop_after_epochs = kwargs.get('drop_after_epochs',3) 
        self.epochs = kwargs.get('epochs',100) 

        print("CREATING MODEL WITH : ",str(kwargs))

        #Placeholders
        self.X = tf.placeholder(tf.float32,[None,self.input_size],name="INPUT")
        self.Y = tf.placeholder(tf.float32,[None,self.output_size],name="Labels")
        self.p_keep_hidden = tf.placeholder("float",name="hidden-net-dropout")
        self.learning_rate = tf.placeholder(tf.float32,shape=(),name="learning-rate")


        l1 = self.__feedforward(self.X,self.input_size,
                                self.hidden_layer_size,
                                activation=tf.nn.leaky_relu,
                                p = self.p_keep_hidden)

        self.model = self.__feedforward(l1,
                                        self.hidden_layer_size,
                                        self.output_size,
                                        activation=None,
                                        bias=0.0)

        with tf.name_scope('Prediction'):
            self.pred = tf.argmax(self.model,1)

        with tf.name_scope('Accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.model,1),tf.argmax(self.Y,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
            tf.summary.scalar('Accuracy',self.accuracy)
        
        with tf.name_scope('Loss'):
            self.cost  = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=self.Y,logits=self.model))
            tf.summary.scalar('Loss', self.cost)
            
        with tf.name_scope('Train'):    
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
            tf.summary.scalar('Learning_Rate', self.learning_rate)

    def __step_learning_rate(self,epoch):
        return self.initial_learning_rate* math.pow(self.drop_percent,
                                            (math.floor((1+epoch)/self.drop_after_epochs)))

    def train(self,trX,trY,teX,teY,NUM_FOLDS=10,BATCH_SIZE=256,TEST_SIZE=512,name="net"):

        test_acc = 0
        acc_mean_accuracy = []
        acc_stdev_accuracy = []

        kf = KFold(n_splits=10,shuffle=True)

        with tf.Session() as session:
            acc_list = []
            time_list = []
            lr_list   = []
            tf.global_variables_initializer().run()
            merged_sum = tf.summary.merge_all()
            writer = tf.summary.FileWriter('logs/{0}_{1}'.format(datetime.now().
                                            strftime("%Y-%m-%d %H:%M:%S"),name), session.graph)

            for step in range(self.epochs):
                print("Epoc : {0}".format(step))
                acc_i  = []
                cost_i = []
                
                for x_t, y_t in kf.split(trX):
                    # train set
                    train_x = np.take(trX,x_t,axis=0)
                    train_y = np.take(trY,x_t,axis=0)

                    # test set
                    test__x = np.take(trX,y_t,axis=0)
                    test__y = np.take(trY,y_t,axis=0)

                    # "single" batch train
                    for i in range(0,len(train_x),BATCH_SIZE):
                        session.run(self.train_op,feed_dict={ 
                                            self.X:train_x[i*BATCH_SIZE:(i+1)*BATCH_SIZE],
                                            self.Y:train_y[i*BATCH_SIZE:(i+1)*BATCH_SIZE],
                                            self.p_keep_hidden:0.5,
                                            self.learning_rate:self.__step_learning_rate(step)})

                test_indices = np.arange(len(teX)) # Get A Test Batch
                np.random.shuffle(test_indices)
                test_indices = test_indices#[0:TEST_SIZE]

                start = time.time()
                acc  = session.run(self.accuracy,feed_dict={
                                        self.X: teX[test_indices],
                                        self.Y: teY[test_indices],
                                        self.p_keep_hidden:1.0,
                                        self.learning_rate:self.__step_learning_rate(step)})
                end   = time.time()
                acc_list.append(acc)
                lr_list.append(self.__step_learning_rate(step))

                #time per prediction
                time_list.append((end-start)/len(test_indices))

                print(step,self.__step_learning_rate(step),acc)

                s = session.run(merged_sum,feed_dict={
                                                self.X: teX[test_indices],
                                                self.Y: teY[test_indices],
                                                self.p_keep_hidden:1.0,
                                                self.learning_rate:self.__step_learning_rate(step)})
                writer.add_summary(s,step)


        return acc_list,time_list,lr_list

    def __feedforward(self,input_t,channels_in,channels_out,activation=None,bias=1.1,p=None):
        with tf.name_scope('FeedForward'):
            w = tf.Variable(tf.random_normal([channels_in,channels_out],stddev=0.1),name="weights")
            b = tf.constant(0.1,tf.float32,[channels_out],name="bias_vector")
            
            if activation == None:
                a = tf.matmul(input_t,w)+b
            else:
                a = activation(tf.matmul(input_t,w)+b)

            if not p is None:
                a = tf.nn.dropout(a,p)

            
            tf.summary.histogram('weights',w)
            tf.summary.histogram('bias_vector',b)

            return a

def main():
    print("Importing Data")
    ((trX, trY),(teX, teY)) = extractor(file='data/emails.csv').min_max_nomalized()
    print(trX[0])
    print("Creating Model")
    model = FFNN()
    print("Training Model")
    print(type(trX))
    acc , t= model.train(trX[0:100],trY[0:100],teX[0:100],teY[0:100])
    print(acc)
    return acc, t
#main()
