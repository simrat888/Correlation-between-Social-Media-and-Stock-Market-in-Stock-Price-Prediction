from keras.layers import LSTM
from keras.layers.core import Activation , Dense , Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop

class LSTMPredictor :
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
		lstm_model = Sequential( )
		length = len( self.__layers )
		for layer in self.__layers :
			length -= 1
			if length != 0 :
				lstm_model.add(
					LSTM( units = layer , return_sequences = True )
					)
			else :
				lstm_model.add( LSTM( layer ) )
			
			lstm_model.add( Dropout( 0.2 ) )
		
		lstm_model.add( Dense( units = 1 ) )
		lstm_model.add( Activation( "linear" ) )
		
		optimizer = RMSprop( self.__learning_rate )
		
		lstm_model.compile(
			loss = self.__loss , optimizer = optimizer ,
			metrics = self.__metrics
			)
		return lstm_model
	
	def train( self , x_train , y_train ) :
		history = self.__model.fit(
				x_train ,
				y_train ,
				batch_size = self.__batch_size ,
				epochs = self.__epochs ,
				validation_split = self.__validation_split
				)
		return history , self.__model
