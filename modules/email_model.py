import numpy as np
import tensorflow as tf
from data_extractor import extractor 
from sklearn.model_selection import KFold

class FFNN():
    def __init__(self,**kwargs):
        args ={
            'learning_rate': 1e-5,
            'hidden_layer_size': 15,
            'input_size': 33485,
            'output_size': 2
        }
        for key, val in kwargs.items():
            args[key] = val

        print("CREATING MODEL WITH : ",args)

        #Placeholders
        self.X = tf.placeholder(tf.float32,[None,args['input_size']],name="INPUT")
        self.Y = tf.placeholder(tf.float32,[None,args['output_size']],name="Labels")


        l1 = self.__feedforward(self.X,args['input_size'],args['hidden_layer_size'],activation=tf.nn.leaky_relu)
        self.model = self.__feedforward(l1,args['hidden_layer_size'],args['output_size'],activation=None,bias=0.0)

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
            self.train_op = tf.train.AdamOptimizer(args['learning_rate']).minimize(self.cost)

    def train(self,trX,trY,teX,teY,NUM_FOLDS=10,BATCH_SIZE=256,TEST_SIZE=512):

        test_acc = 0
        acc_mean_accuracy = []
        acc_stdev_accuracy = []

        kf = KFold(n_splits=10,shuffle=True)

        with tf.Session() as session:
            tf.global_variables_initializer().run()

            for step in range(100):
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
                                            self.Y:train_y[i*BATCH_SIZE:(i+1)*BATCH_SIZE]})

                test_indices = np.arange(len(teX)) # Get A Test Batch
                np.random.shuffle(test_indices)
                test_indices = test_indices[0:TEST_SIZE]

                print(step,session.run(self.accuracy,feed_dict={
                                                    self.X: teX[test_indices],
                                                    self.Y: teY[test_indices]}))

    def __feedforward(self,input_t,channels_in,channels_out,activation=None,bias=0.1):
        with tf.name_scope('FeedForward'):
            w = tf.Variable(tf.random_normal([channels_in,channels_out],stddev=0.1),name="weights")
            b = tf.constant(0.1,tf.float32,[channels_out],name="bias_vector")
            
            if activation == None:
                a = tf.matmul(input_t,w)+b
            else:
                a = activation(tf.matmul(input_t,w)+b)
            
            tf.summary.histogram('weights',w)
            tf.summary.histogram('bias_vector',b)

            return a

def main():
    print("Importing Data")
    ((trX, trY),(teX, teY)) = extractor(file='../data/emails.csv').nomalized()
    print("Creating Model")
    model = FFNN()
    print("Training Model")
    print(type(trX))
    model.train(trX,trY,teX,teY)
main()
