from keras.layers import GRU
from keras.layers.core import Activation , Dense , Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop

class GRUPredictor :
	def __init__(
			self , layers , learning_rate = 0.0001 , loss = "mse" ,
			metrics = [ 'mae' , 'mse' ] , batch_size = 512 , epochs = 100 ,
			validation_split = 0.05
			) :
		self.__layers = layers
		self.__metrics = metrics
		self.__learning_rate = learning_rate
		self.__loss = loss
		self.__batch_size = batch_size
		self.__epochs = epochs
		self.__validation_split = validation_split
		self.__model = self.__build_model( )
	
	def __build_model( self ) :
		gru_model = Sequential( )
		
		length = len( self.__layers )
		for layer in self.__layers :
			length -= 1
			if length != 0 :
				gru_model.add( GRU( units = layer , return_sequences = True ) )
			else :
				gru_model.add( GRU( layer ) )
			
			gru_model.add( Dropout( 0.2 ) )
		
		gru_model.add( Dense( units = 1 ) )
		gru_model.add( Activation( "linear" ) )
		
		optimizer = RMSprop( self.__learning_rate )
		
		gru_model.compile(
			loss = self.__loss , optimizer = optimizer ,
			metrics = self.__metrics
			)
		return gru_model
	
	def train( self , x_train , y_train ) :
		history = self.__model.fit(
				x_train ,
				y_train ,
				batch_size = self.__batch_size ,
				epochs = self.__epochs ,
				validation_split = self.__validation_split
				)
		return history , self.__model
