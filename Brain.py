import tensorflow as tf
import time, threading
from datetime import datetime

from keras.models import *
from keras.layers import *
from keras import backend as K

class Brain:
    train_queue = [ [], [], [], [], [] ]	# s, a, r, s', s' terminal mask
    lock_queue = threading.Lock()

    def __init__(self, config):
        sessionConfig = tf.ConfigProto()
        sessionConfig.gpu_options.allow_growth=True
        #sessionConfig.log_device_placement = True
        self.session = tf.Session(config=sessionConfig)
        self.BatchCounter = 0
        self.Config = config

        K.set_session(self.session)
        K.manual_variable_initialization(True)

        self.model = self._build_model()
        self.graph = self._build_graph(self.model)

        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()

        self.WriterMergedSummary = tf.summary.merge_all()
        self.Writer = tf.summary.FileWriter("logs/Run" + datetime.now().strftime("%Y-%m-%d_%H-%M"), self.session.graph)

        self.default_graph.finalize()	# avoid modifications, make the TF graph read only so no other threads can chcange the ops

    def _build_model(self):

        l_input = Input( batch_shape=(None, self.Config.ObservationDimentions[0], self.Config.ObservationDimentions[1], self.Config.ObservationDimentions[2]))

        conv1 = Conv2D(filters = 32, kernel_size = 8,strides =4, padding='valid', activation='relu', name='Layer_CONV1')(l_input)
        conv2 = Conv2D(filters = 64, kernel_size = 4,strides =2, padding='valid', activation='relu', name='Layer_CONV2')(conv1)
        conv3 = Conv2D(filters = 64, kernel_size = 3,strides =1, padding='valid', activation='relu', name='Layer_CONV3')(conv2)
        conv4 = Conv2D(filters = 32, kernel_size = 7,strides =1, padding='valid', activation='relu', name='Layer_CONV4')(conv3)

        l_dense1 = Dense(512, activation='relu')(conv4)
        l_dense2 = Dense(16, activation='relu')(l_dense1)

        out_actions = Dense(self.Config.ActionsCount, activation='softmax', name='Actions')(l_dense2)
        out_value   = Dense(1, activation='linear', name='ActionValueFunctionEstimation')(l_dense2)

        model = Model(inputs=[l_input], outputs=[out_actions, out_value])
        model._make_predict_function()	# have to initialize before threading
        
        self.Trainable = tf.trainable_variables()

        return model

    def _build_graph(self, model):

        # State tensor        
        s_t = tf.placeholder(tf.float32, shape=(None, self.Config.ObservationDimentions[0], self.Config.ObservationDimentions[1], self.Config.ObservationDimentions[2]), name='S_T')

        # Acton tensor
        a_t = tf.placeholder(tf.float32, shape=(None, self.Config.ActionsCount), name='A_T')

        # Reward tensor
        r_t = tf.placeholder(tf.float32, shape=(None, 1), name='R_T') # not immediate, but discounted n step reward

        # Policy and Action-Value function estimation
        p, v = model(s_t)

        # Calculate the logged propability of action for the loss function (plus add small number to avoid NaN)
        log_prob = tf.log( tf.reduce_sum(p * a_t, axis=1, keep_dims=True) + 1e-10)

        # Calculate the advantage function value (aproximation of the A(s,a))
        advantage = r_t - v

        # Loss function calculation
        # Loss function  L = loss_policy + loss_value + entropy
        loss_policy = - log_prob * tf.stop_gradient(advantage)									# maximize policy

        loss_value  = self.Config.LossFuctionValueCoefficient * tf.square(advantage)												# minimize value error
        entropy = self.Config.LossFunctionEntropy * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1, keep_dims=True)	# maximize entropy (regularization)

        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

        optimizer = tf.train.RMSPropOptimizer(self.Config.LearningRate, decay=.99)
        #optimizer = tf.train.AdamOptimizer(self.Config.LearningRate)
        minimize = optimizer.minimize(loss_total)
        
        # set some histograms up
        with tf.name_scope('Training_parameters'):
            #tf.summary.scalar("Advantage", advantage)
            #tf.summary.histogram("Logged Propability of Action", log_prob)
            tf.summary.histogram("Loss_Value", loss_value)
            tf.summary.histogram("Loss_Policy", loss_policy)
            tf.summary.histogram("Entropy", entropy)
            tf.summary.histogram("Total_Loss", loss_policy + loss_value + entropy)

        return s_t, a_t, r_t, minimize

    def Optimize(self, id):

        if len(self.train_queue[0]) < self.Config.MiniBatchSize:
            time.sleep(0)	# yield
            return

        with self.lock_queue:
            if len(self.train_queue[0]) < self.Config.MiniBatchSize:	# more thread could have passed without lock
                return 									# we can't yield inside lock

            print(datetime.now(), "Optimizer:", id, " -> ", "Neural network training session started....")

            s, a, r, s_, s_mask = self.train_queue
            self.train_queue = [ [], [], [], [], [] ]
    
        s = np.reshape(s, [-1, self.Config.ObservationDimentions[0], self.Config.ObservationDimentions[1], self.Config.ObservationDimentions[2]])
        a = np.reshape(a, [-1, self.Config.ActionsCount])
        r = np.reshape(r, [-1, 1])
        s_ = np.reshape(s_, [-1, self.Config.ObservationDimentions[0], self.Config.ObservationDimentions[1], self.Config.ObservationDimentions[2]])
        s_mask = np.reshape(s_mask, [-1,1])

        if len(s) > 5* self.Config.MiniBatchSize: print(datetime.now(), "Optimizer alert! Minimizing batch of %d" % len(s))

        v = self.predict_v(s_)
        r = r + self.Config.N_StepRewardDiscount * np.reshape(v, [-1, 1]) * s_mask	# set v to 0 where s_ is terminal state

        s_t, a_t, r_t, minimize = self.graph
        result = self.session.run(minimize, feed_dict={s_t: s, a_t: a, r_t: r})

        summary = self.session.run(self.WriterMergedSummary, feed_dict={s_t: s, a_t: a, r_t: r})
        self.BatchCounter+=1
        self.Writer.add_summary(summary, self.BatchCounter)

        print(datetime.now(), "Optimizer:", id, " -> ", "Neural network training session completed....")

    def train_push(self, s, a, r, s_):
        with self.lock_queue:
            self.train_queue[0].append(s)
            self.train_queue[1].append(a)
            self.train_queue[2].append(r)

            if s_ is None:
                self.train_queue[3].append(self.Config.TerminateState)
                self.train_queue[4].append(0.)
            else:	
                self.train_queue[3].append(s_)
                self.train_queue[4].append(1.)

    def predict(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return p, v

    def predict_p(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)		
            return p

    def predict_v(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)		
            return v

    def AddSummary(self, tagName, index, value, group):        
        self.Writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag=tagName, simple_value=value)]), index)
        #self.Writer.add_summary(tf.Summary(value=[tf.summary.scalar(tagName, value, collections=[group])]), index)

    def AddHistogramSummary(self, tagName, index, value):
        self.Writer.add_summary(tf.Summary(tf.summary.histogram(tagName, value)))

    def AddMultipleValueSummary(self, tagName, index, values):
        self.Writer.add_summary(tf.Summary(value=[tf.summary.tensor_summary(tag=tagName, tensor=values)]), index)

    def CloseSession(self):
        self.Session.close()