import os

import matplotlib.pyplot as plt
import pandas as pd
from yahoo_fin import stock_info as si

class YahooStockData :
	def __init__(
			self , company , from_date , to_date , save_folder = "charts" ,
			result_folder = "results" , interval = '1d' ,
			stock_df = None
			) :
		self.__ticker = company
		self.__start_date = from_date
		self.__end_date = to_date
		self.interval = interval
		self.__result_folder = f"results/{result_folder}"
		self.__stock_df = stock_df
		self.__save_chart_folder = f"{self.__result_folder}/{save_folder}"
		self.__create_folder( self.__save_chart_folder )
	
	def __create_folder( self , folder ) :
		if not os.path.exists( folder ) :
			os.makedirs( folder )
	
	def get_stock_data( self ) :
		__stock_data = si.get_data(
			self.__ticker , start_date = self.__start_date ,
			end_date = self.__end_date ,
			interval = self.interval
			)
		self.__stock_df = pd.DataFrame( __stock_data.to_records( ) ).rename(
			columns = { 'index' : 'date' }
			)
		self.__stock_df = self.__stock_df.set_index( "date" )
		self.__stock_df.sort_index( ascending = True )
		return self.__stock_df
	
	def plot_stock_data(
			self , column = 'adjclose' , style = 'seaborn-dark' ,
			figsize = (10 , 15)
			) :
		if self.__stock_df is None :
			self.get_stock_data( )
		plt.style.use( style )
		self.__stock_df[ column ].plot( cmap = "viridis" , figsize = figsize )
		plt.grid( )
		file_path = os.path.join(
			self.__save_chart_folder , f'{self.__ticker}_stock_data_.png'
			)
		plt.savefig( file_path )
		plt.show( )
		plt.close( )
	
	def save_stock_data_as_pkl( self , file_name , folder = "data_sets" ) :
		file_path = f"{folder}/{file_name}"
		if self.__stock_df is None :
			self.get_stock_data( )
		
		if not os.path.exists( os.path.dirname( file_path ) ) :
			os.mkdir( os.path.dirname( file_path ) )
		
		self.__stock_df.to_pickle( file_path )
		return file_path
	
	def load_stock_data_from_pkl( self , file_path ) :
		self.__stock_df = pd.read_pickle( file_path )
		return self.__stock_df
