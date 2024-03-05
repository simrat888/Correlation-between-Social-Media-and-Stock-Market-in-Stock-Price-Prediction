import json
import os

import matplotlib.pyplot as plt
import numpy as np

results_folder = "results"

sub_folders = [ f.path for f in os.scandir( results_folder ) if f.is_dir( ) ]

rmse_values = {
		"Random Forest" : [ ] , "LSTM" : [ ] , "GRU" : [ ] , "CNN" : [ ]
		}
epochs_list = [ ]
seq_lengths_list = [ ]

def get_metrics( ) :
	for sub_folder in sub_folders :
		for file in os.listdir( sub_folder ) :
			if file.endswith( ".json" ) :
				results_file = os.path.join( sub_folder , file )
				with open( results_file , "r" ) as f :
					results_data = json.load( f )
				
				for model in [ "Random Forest" , "LSTM" , "GRU" , "CNN" ] :
					rmse_values[ model ].append(
							results_data[ "all_models" ][ "rmse" ][ model ]
							)
				
				epochs = results_data[ "train_metadata" ][ "epochs" ]
				seq_len = results_data[ "train_metadata" ][ "seq_len" ]
				epochs_list.append( epochs )
				seq_lengths_list.append( seq_len )

def plot_and_save_metric_and_save( ) :
	bar_width = 0.15
	x = np.arange( len( epochs_list ) )
	fig , ax = plt.subplots( figsize = (12 , 6) )
	ax.bar(
			x - 3 * bar_width / 2 , rmse_values[ "Random Forest" ] ,
			bar_width ,
			label = "Random Forest"
			)
	ax.bar(
			x - bar_width / 2 , rmse_values[ "LSTM" ] , bar_width ,
			label = "LSTM"
			)
	ax.bar(
			x + bar_width / 2 , rmse_values[ "GRU" ] , bar_width , label =
			"GRU"
			)
	ax.bar(
			x + 3 * bar_width / 2 , rmse_values[ "CNN" ] , bar_width , label =
			"CNN"
			)
	ax.set_ylabel( "RMSE" )
	ax.set_title(
			"RMSE Comparison for Different Models, Epochs, and Sequence "
			"Lengths"
			)
	ax.set_xticks( x )
	ax.set_xticklabels(
			[ f"E{e}, S{s}" for e , s in
			  zip( epochs_list , seq_lengths_list ) ] , rotation = 45
			)
	ax.legend( )
	plt.savefig( "rmse_comparison_chart_epochs_seq_lengths_bar.png" )
	plt.show( )

def get_final_good_model( ) :
	global best_worst_rmse
	best_worst_rmse = {
			"Random Forest" : { "best" : float( "inf" ) } ,
			"LSTM"          : { "best" : float( "inf" ) , "worst" : 0 } ,
			"GRU"           : { "best" : float( "inf" ) , "worst" : 0 } ,
			"CNN"           : { "best" : float( "inf" ) , "worst" : 0 } ,
			}
	
	for model in [ "Random Forest" , "LSTM" , "GRU" , "CNN" ] :
		for i , rmse_value in enumerate( rmse_values[ model ] ) :
			if model == "Random Forest" :
				best_worst_rmse[ model ][ "best" ] = rmse_value
			else :
				if rmse_value < best_worst_rmse[ model ][ "best" ] :
					best_worst_rmse[ model ][ "best" ] = rmse_value
					best_worst_rmse[ model ][ "best_seq_len" ] = \
						seq_lengths_list[ i ]
					best_worst_rmse[ model ][ "best_epochs" ] = epochs_list[
						i ]
				if rmse_value > best_worst_rmse[ model ][ "worst" ] :
					best_worst_rmse[ model ][ "worst" ] = rmse_value
					best_worst_rmse[ model ][ "worst_seq_len" ] = \
						seq_lengths_list[ i ]
					best_worst_rmse[ model ][ "worst_epochs" ] = epochs_list[
						i ]
	
	final_best_model = min(
			best_worst_rmse , key = lambda x : best_worst_rmse[ x ][ "best" ]
			)
	best_worst_rmse[ "final_best_model" ] = final_best_model
	
	with open( "best_worst_rmse_final_good_model.json" , "w" ) as f :
		json.dump( best_worst_rmse , f , indent = 2 )
