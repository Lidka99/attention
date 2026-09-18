
import tensorflow as tf
import numpy as np

class AttentiveConv2D(tf.keras.layers.Layer):
    def __init__(self,filters,kernel_size,activation,**kwargs):
        super().__init__(**kwargs)
        self.filters=filters
        self.kernel_size=kernel_size
        self.activation=activation
        self.conv2d=tf.keras.layers.Conv2D(filters=self.filters,kernel_size=self.kernel_size,activation=self.activation,strides=(1,1),padding='valid')

    def build(self,input_shape):
        self.attention_weights=self.add_weight(name='attention',shape=(self.filters,),initializer=tf.keras.initializers.Constant(1.0),trainable=True)
        super().build(input_shape)

    def call(self,inputs,use_attention=True):
        outputs=self.conv2d(inputs)
        attention=tf.nn.relu(self.attention_weights)
        outputs=outputs*attention
        return [outputs,attention]

    def get_config(self):
        config=super().get_config()
        config.update({'filters':self.filters,'kernel_size':self.kernel_size,'activation':self.activation})
        return config
        
def l1_loss(y_true,y_pred):
    return tf.reduce_mean(tf.abs(y_pred))
        
def get_model():
    # create model
    x=tf.keras.layers.Input((28,28,1))
    # 1
    t,o1=AttentiveConv2D(filters=64,kernel_size=3,activation='relu')(x)
    t=tf.keras.layers.MaxPooling2D(pool_size=(2,2))(t)
    # 2
    t,o2=AttentiveConv2D(filters=48,kernel_size=3,activation='relu')(t)
    t=tf.keras.layers.MaxPooling2D(pool_size=(2,2))(t)
    # 3
    t,o3=AttentiveConv2D(filters=32,kernel_size=3,activation='relu')(t)
    t=tf.keras.layers.MaxPooling2D(pool_size=(2,2))(t)
    # Classifier
    t=tf.keras.layers.Flatten()(t)
    t=tf.keras.layers.Dense(32,activation='relu')(t)
    y=tf.keras.layers.Dense(10,activation='softmax')(t)
    # Model
    model=tf.keras.Model(inputs=[x],outputs=[y,o1,o2,o3],name='test_model')
    model.summary()
    return model
    
# load MNIST
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()#tf.keras.datasets.fashion_mnist.load_data()#tf.keras.datasets.mnist.load_data()
x_train=np.reshape(x_train,(len(x_train),28,28,1)).astype('float32')/255.0-0.5
x_test=np.reshape(x_test,(len(x_test),28,28,1)).astype('float32')/255.0-0.5
y_train=tf.keras.utils.to_categorical(y_train,10)
y_test=tf.keras.utils.to_categorical(y_test,10)
# training
z1=tf.zeros((len(x_train),64),dtype=np.float32)
z2=tf.zeros((len(x_train),48),dtype=np.float32)
z3=tf.zeros((len(x_train),32),dtype=np.float32)
for beta in [0.0,0.1,0.2,0.5,1.0]:
    print('#---------------------------------------------------')
    print('Beta:',beta)
    model=get_model()
    # 1
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,loss=[tf.keras.losses.CategoricalCrossentropy(),l1_loss,l1_loss,l1_loss,],loss_weights=[1.0,beta,beta,beta],
        metrics=[tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.MeanAbsoluteError(),tf.keras.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanAbsoluteError()])
    model.fit(x_train,[y_train,z1,z2,z3],epochs=20)
    # 2
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,loss=[tf.keras.losses.CategoricalCrossentropy(),l1_loss,l1_loss,l1_loss,],loss_weights=[1.0,beta,beta,beta],
        metrics=[tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.MeanAbsoluteError(),tf.keras.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanAbsoluteError()])
    model.fit(x_train,[y_train,z1,z2,z3],epochs=20)
    # 3 tuning process
    model.layers[1].attention_weights._trainable=False
    model.layers[3].attention_weights._trainable=False
    model.layers[5].attention_weights._trainable=False
    model.summary()
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,loss=[tf.keras.losses.CategoricalCrossentropy(),l1_loss,l1_loss,l1_loss,],loss_weights=[1.0,0.0,0.0,0.0],
        metrics=[tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.MeanAbsoluteError(),tf.keras.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanAbsoluteError()])
    model.fit(x_train,[y_train,z1,z2,z3],epochs=100)
    model.save_weights('weights_beta_'+str(beta)+'.h5')
    # evaluate
    z1=tf.zeros((len(x_test),48),dtype=np.float32)
    z2=tf.zeros((len(x_test),32),dtype=np.float32)
    z3=tf.zeros((len(x_test),32),dtype=np.float32)
    h=model.evaluate(x_test,[y_test,z1,z2,z3])
    w1=tf.nn.relu(model.layers[1].get_weights()[0])
    w2=tf.nn.relu(model.layers[3].get_weights()[0])
    w3=tf.nn.relu(model.layers[5].get_weights()[0])
    print('Layer 1:',np.count_nonzero(w1==0))
    print('Layer 2:',np.count_nonzero(w2==0))
    print('Layer 3:',np.count_nonzero(w3==0))
    print('#---------------------------------------------------')