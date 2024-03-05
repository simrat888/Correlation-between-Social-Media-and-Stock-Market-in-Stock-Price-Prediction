import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error , mean_squared_error

class PerformanceUtil :
	def __init__(
			self , histories , all_predictions , y_test , metadata ,
			save_folder = "charts" , result_folder = "" ,
			model_folder = 'models'
			) :
		self.__histories = histories
		self.__all_predictions = all_predictions
		self.__y_test = y_test
		self.__result_folder = f"results/{result_folder}"
		self.__save_chart_folder = f"{self.__result_folder}/{save_folder}"
		self.__save_model_folder = f"{self.__result_folder}/{model_folder}"
		self.__metadata = metadata
		self.__error_metrics = None
		self.__create_folder( self.__save_model_folder )
		self.__create_folder( self.__save_chart_folder )
	
	def __create_folder( self , folder ) :
		if not os.path.exists( folder ) :
			os.makedirs( folder )
	
	def __root_absolute_squared_error(
			self , y_true , y_pred , threshold = 0.5
			) :
		return np.sqrt( mean_absolute_error( y_true , y_pred ) )
	
	def __root_mean_squared_error( self , y_true , y_pred ) :
		return np.sqrt( mean_squared_error( y_true , y_pred ) )
	
	def __get_error_metrics( self , y_test ) :
		performance_metrics = [ 'rmse' , 'mae' ]
		self.__error_metrics = { metric : { } for metric in
		                         performance_metrics }
		for model_name , predictions in self.__all_predictions.items( ) :
			self.__error_metrics[ 'rmse' ][
				model_name ] = self.__root_mean_squared_error(
				y_test , predictions
				)
			self.__error_metrics[ 'mae' ][
				model_name ] = self.__root_absolute_squared_error(
				y_test , predictions
				)
	
	def __write_result( self , best_model_name ) :
		data = {
				"train_metadata" : self.__metadata ,
				"all_models"     : self.__error_metrics ,
				"best_model"     : best_model_name
				}
		model_file_path = os.path.join(
			self.__result_folder , "model_metrics_with_best_models.json"
			)
		
		with open( model_file_path , "w" ) as file :
			json.dump( data , file , indent = 4 )
	
	def save_models( self , models ) :
		for model_name , model in models.items( ) :
			model_file_path = os.path.join(
				self.__save_model_folder , f'{model_name}_model.pkl'
				)
			with open( model_file_path , 'wb' ) as f :
				pickle.dump( model , f )
	
	def print_and_writebest_model_based_on_rmse( self ) :
		best_model_name = min(
				self.__error_metrics[ 'rmse' ] ,
				key = self.__error_metrics[ 'rmse' ].get
				)
		best_rmse = self.__error_metrics[ 'rmse' ][ best_model_name ]
		print( "All Models and Root Mean Squared Error:" )
		for model_name , rmse in self.__error_metrics[ 'rmse' ].items( ) :
			print( f"\t{model_name}'s RMSE is {rmse:.4f}" )
		
		print(
			f"The best model is {best_model_name} with an RMSE of "
			f"{best_rmse:.4f}"
			)
		self.__write_result( best_model_name )
	
	def plot_loss_curves( self ) :
		num_models = len( self.__histories )
		fig , axs = plt.subplots(
			num_models , 1 , figsize = (10 , num_models * 5) ,
			constrained_layout = True
			)
		
		for i , model_history in enumerate( self.__histories ) :
			for model_name , history in model_history.items( ) :
				axs[ i ].plot(
						history.history[ 'loss' ] , label = 'Training Loss'
						)
				axs[ i ].plot(
						history.history[ 'val_loss' ] ,
						label = 'Validation Loss'
						)
				axs[ i ].plot( history.history[ 'mse' ] , label = "MSE" )
				axs[ i ].plot( history.history[ 'mae' ] , label = "MAE" )
				axs[ i ].set_title( f'Metrics of {model_name}' )
				axs[ i ].set_xlabel( 'Epoch' )
				axs[ i ].set_ylabel( 'Loss' )
				axs[ i ].legend( loc = 'upper right' )
		file_path = os.path.join(
			self.__save_chart_folder , 'all_model_loss_curves.png'
			)
		plt.savefig( file_path )
		plt.show( )
		plt.close( )
	
	def plot_all_models_performance( self ) :
		for model_history in self.__histories :
			for model_name , history in model_history.items( ) :
				plt.plot(
						history.history[ 'val_loss' ] ,
						label = f"{model_name} Validation"
						)
		plt.title( 'All Models Validation Loss' )
		plt.xlabel( 'Epoch' )
		plt.ylabel( 'Loss' )
		plt.legend( loc = 'upper right' )
		file_path = os.path.join(
			self.__save_chart_folder , 'all_model_validation_loss_curves.png'
			)
		plt.savefig( file_path )
		plt.show( )
		plt.close( )
	
	def plot_all_predictions( self ) :
		plt.plot( self.__y_test , label = 'True Values' )
		for model_name , predictions in self.__all_predictions.items( ) :
			plt.plot( predictions , label = f'{model_name} Predictions' )
		plt.title( 'All Models Predictions vs True Values' )
		plt.legend( loc = 'upper right' )
		plt.show( )
		plt.close( )
		
		n_models = len( self.__all_predictions )
		fig , axes = plt.subplots(
			n_models , 1 , figsize = (10 , n_models * 5) , sharex = True
			)
		
		for i , (model_name , predictions) in enumerate(
				self.__all_predictions.items( )
				) :
			axes[ i ].plot(
				self.__y_test , label = 'True Values' , alpha = 0.6
				)
			axes[ i ].plot(
				predictions , label = f'{model_name} Predictions' , alpha = 0.6
				)
			axes[ i ].set_title( f'{model_name} Predictions vs True Values' )
			axes[ i ].legend( loc = 'upper right' )
		
		plt.xlabel( 'Time Steps' )
		plt.tight_layout( )
		file_path = os.path.join(
			self.__save_chart_folder ,
			'all_model_predictions_vs_true_values.png'
			)
		plt.savefig( file_path )
		plt.show( )
		plt.close( )
	
	def plot_error_metric_comparison( self ) :
		self.__get_error_metrics( self.__y_test )
		
		fig , axs = plt.subplots( 1 , 2 , figsize = (15 , 5) )
		
		for idx , (metric , model_results) in enumerate(
				self.__error_metrics.items( )
				) :
			model_names = list( model_results.keys( ) )
			metric_values = list( model_results.values( ) )
			
			axs[ idx ].bar( model_names , metric_values )
			axs[ idx ].set_title( f'{metric.upper( )} Comparison' )
			axs[ idx ].set_xlabel( 'Models' )
			axs[ idx ].set_ylabel( metric.upper( ) )
		
		file_path = os.path.join(
			self.__save_chart_folder , 'all_model_error_metric_comparison.png'
			)
		plt.savefig( file_path )
		plt.show( )
		plt.close( )
