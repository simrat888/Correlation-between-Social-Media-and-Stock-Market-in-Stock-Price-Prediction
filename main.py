import datetime

from Stock_pred_ML_fin.CNNPredictor import CNNPredictor
from Stock_pred_ML_fin.GRUPredictor import GRUPredictor
from Stock_pred_ML_fin.LSTMPredictor import LSTMPredictor
from Stock_pred_ML_fin.PerformanceUtil import PerformanceUtil
from Stock_pred_ML_fin.RandomForestPredictor import RandomForestPredictor
from Stock_pred_ML_fin.StockDataPreprocessor import StockDataPreprocessor
from Stock_pred_ML_fin.YahooStockData import YahooStockData

def train_and_evaluate_models(
		x_train , y_train , x_test , y_test , layers , epochs
		) :
	model_histories = [ ]
	
	rf_predictor = RandomForestPredictor( )
	rf_history , rf_model = rf_predictor.train( x_train , y_train )
	
	lstm_predictor = LSTMPredictor( layers , epochs = epochs )
	lstm_history , lstm_model = lstm_predictor.train( x_train , y_train )
	model_histories.append( { "LSTM" : lstm_history } )
	
	gru_predictor = GRUPredictor( layers , epochs = epochs )
	gru_history , gru_model = gru_predictor.train( x_train , y_train )
	model_histories.append( { "GRU" : gru_history } )
	
	cnn_predictor = CNNPredictor(
		layers , input_shape = (x_train.shape[ 1 ] , 1) , epochs = epochs
		)
	cnn_history , cnn_model = cnn_predictor.train( x_train , y_train )
	model_histories.append( { "CNN" : cnn_history } )
	
	all_predictions = {
			"Random Forest" : rf_model.predict(
				x_test.reshape( x_test.shape[ 0 ] , -1 )
				) ,
			"LSTM"          : lstm_model.predict( x_test ) ,
			"GRU"           : gru_model.predict( x_test ) ,
			"CNN"           : cnn_model.predict( x_test )
			}
	
	all_models = {
			"Random Forest" : rf_model ,
			"LSTM"          : lstm_model ,
			"GRU"           : gru_model ,
			"CNN"           : cnn_model
			}
	
	return model_histories , all_predictions , all_models

def get_and_save_yahoo_stock_data(
		ticker , start_date , end_date , result_folder , file_name
		) :
	yahoo_stock_data = YahooStockData(
		ticker , start_date , end_date , result_folder = result_folder
		)
	stock_df = yahoo_stock_data.get_stock_data( )
	yahoo_stock_data.plot_stock_data( )
	file_path = yahoo_stock_data.save_stock_data_as_pkl( file_name )
	stock_df_loaded = yahoo_stock_data.load_stock_data_from_pkl( file_path )
	return stock_df_loaded

def main( ) :
	start_date = '1875-02-01 00:00:00'
	end_date = '2023-04-01 23:59:59'
	ticker = "AAPL"
	file_name = "apple.pkl"
	
	seq_len = 15
	epochs = 200
	layers = [ seq_len , 50 , 100 , 1 ]
	
	current_time = datetime.datetime.now( )
	time_stamp = current_time.strftime( "%d_%H-%M-%S" )
	
	result_folder = f"Ticker_{ticker}_{time_stamp}_layers" + "_".join(
			[ str( i ) for i in layers ]
			) + f"_epochs_{epochs}"
	
	stock_df_loaded = get_and_save_yahoo_stock_data(
		ticker , start_date , end_date , result_folder , file_name
		)
	
	data_preprocessing = StockDataPreprocessor( seq_len )
	stock_data = stock_df_loaded[ 'adjclose' ].values
	x_train , y_train , x_test , y_test = data_preprocessing.prepare_data(
		stock_data
		)
	
	model_histories , all_predictions , all_models = train_and_evaluate_models(
		x_train , y_train , x_test , y_test , layers ,
		epochs
		)
	
	metadata = {
			"trained_at" : time_stamp ,
			"seq_len"    : seq_len ,
			"epochs"     : epochs ,
			"layers"     : layers ,
			"data"       : {
					"ticker" : ticker , "start_date" : start_date ,
					"end_date" : end_date ,
					"count" : {
							"x_train" : len( x_train ) ,
							"y_train" : len( y_train ) ,
							"x_test"  : len( x_test ) ,
							"y_test"  : len( y_test )
							}
					}
			}
	
	performance_util = PerformanceUtil(
		model_histories , all_predictions , y_test ,
		result_folder = result_folder ,
		metadata = metadata
		)
	performance_util.save_models( all_models )
	performance_util.plot_loss_curves( )
	performance_util.plot_all_models_performance( )
	performance_util.plot_all_predictions( )
	performance_util.plot_error_metric_comparison( )
	performance_util.print_and_writebest_model_based_on_rmse( )
