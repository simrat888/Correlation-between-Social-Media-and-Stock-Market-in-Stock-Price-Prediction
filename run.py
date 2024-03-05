from Stock_pred_ML_fin.final_metrics_plot import (
	get_final_good_model ,
	get_metrics , plot_and_save_metric_and_save ,
	)
from Stock_pred_ML_fin.main import main

if __name__ == "__main__" :
	main( )
	get_metrics( )
	plot_and_save_metric_and_save( )
	get_final_good_model( )
