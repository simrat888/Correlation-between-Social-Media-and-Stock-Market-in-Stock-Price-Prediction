from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error , mean_squared_error
from sklearn.model_selection import train_test_split

class RandomForestPredictor :
	def __init__( self , n_estimators = 100 , random_state = 42 ) :
		self.__n_estimators = n_estimators
		self.__random_state = random_state
		self.__model = RandomForestRegressor(
			n_estimators = self.__n_estimators ,
			random_state = self.__random_state
			)
		self.__history = None
	
	def train( self , x_train , y_train ) :
		x_train , x_val , y_train , y_val = train_test_split(
			x_train , y_train , test_size = 0.2 , random_state = 42
			)
		
		x_train_2d = x_train.reshape( x_train.shape[ 0 ] , -1 )
		x_val_2d = x_val.reshape( x_val.shape[ 0 ] , -1 )
		
		history = {
				'loss' : [ ] , 'val_loss' : [ ] , 'mae' : [ ] , 'mse' : [ ]
				}
		
		self.__model.fit( x_train_2d , y_train )
		__train_predictions = self.__model.predict( x_train_2d )
		__val_predictions = self.__model.predict( x_val_2d )
		
		__train_loss = mean_squared_error( y_train , __train_predictions )
		__val_loss = mean_squared_error( y_val , __val_predictions )
		__train_mae = mean_absolute_error( y_train , __train_predictions )
		
		history[ 'loss' ].append( __train_loss )
		history[ 'val_loss' ].append( __val_loss )
		history[ 'mae' ].append( __train_mae )
		history[ 'mse' ].append( __val_loss )
		
		self.__history = history
		return history , self.__model
