import numpy as np

class StockDataPreprocessor :
	def __init__(
			self , stock_history_length = 50 , split_ratio = 0.8 ,
			normalise_stock_data = True
			) :
		self.__stock_history_length = stock_history_length
		self.__normalise_stock_data = normalise_stock_data
		self.__split_ratio = split_ratio
	
	def __normalise_data( self , data ) :
		normalised_data = [ ]
		for window in data :
			normalised_window = [ ((float( p ) / float( window[ 0 ] )) - 1) for
			                      p in window ]
			normalised_data.append( normalised_window )
		return normalised_data
	
	def __train_test_split( self , data_to_split ) :
		split_index = int( len( data_to_split ) * self.__split_ratio )
		train_data = data_to_split[ :split_index ]
		test_data = data_to_split[ split_index : ]
		return train_data , test_data
	
	def __extract_y_from_data( self , data ) :
		data = np.array( data )
		inp = data[ : , :-1 ]
		out = data[ : , -1 ]
		inp = np.reshape( inp , (inp.shape[ 0 ] , inp.shape[ 1 ] , 1) )
		return inp , out
	
	def prepare_data( self , data ) :
		history_length = self.__stock_history_length + 1
		__nomalise_data = [ ]
		for index in range( len( data ) - history_length ) :
			__nomalise_data.append( data[ index : index + history_length ] )
		
		if self.__normalise_stock_data :
			__nomalise_data = self.__normalise_data( __nomalise_data )
		
		train_data , test_data = self.__train_test_split( __nomalise_data )
		x_train , y_train = self.__extract_y_from_data( train_data )
		x_test , y_test = self.__extract_y_from_data( test_data )
		
		return x_train , y_train , x_test , y_test
