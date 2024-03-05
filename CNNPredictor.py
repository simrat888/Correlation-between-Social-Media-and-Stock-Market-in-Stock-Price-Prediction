from keras.layers import Conv1D , Flatten , MaxPooling1D
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop

class CNNPredictor :
	def __init__(
			self , layers , input_shape , learning_rate = 0.0001 ,
			filters = 64 , kernel_size = 3 , pool_size = 2 , epochs = 100 ,
			loss = "mse" ,
			metrics = [ 'mae' , 'mse' ]
			) :
		self.__layers = layers
		self.__filters = filters
		self.__kernel_size = kernel_size
		self.__pool_size = pool_size
		self.__input_shape = input_shape
		self.__epochs = epochs
		self.__loss = loss
		self.__metrics = metrics
		self.__learning_rate = learning_rate
		self.__model = self.__build_model( )
	
	def __build_model( self ) :
		cnn_model = Sequential( )
		
		cnn_model.add(
			Conv1D(
				filters = self.__filters , kernel_size = self.__kernel_size ,
				activation = 'relu' ,
				input_shape = self.__input_shape
				)
			)
		cnn_model.add( MaxPooling1D( pool_size = self.__pool_size ) )
		cnn_model.add( Flatten( ) )
		
		for units in self.__layers[ 1 : ] :
			cnn_model.add( Dense( units = units , activation = 'relu' ) )
		optimizer = RMSprop( self.__learning_rate )
		
		cnn_model.compile(
			optimizer = optimizer , loss = self.__loss ,
			metrics = self.__metrics
			)
		return cnn_model
	
	def train( self , x_train , y_train ) :
		history = self.__model.fit(
				x_train ,
				y_train ,
				epochs = self.__epochs ,
				validation_split = 0.05
				)
		return history , self.__model
