## Predict Stock Prices Using PYthon.
## This is for research purposes only. Do not use this to back stock buying decisions. 
## I MA NOT RESPONSIBLE FOR ANYTHING! NO JOKE!

# DO THIS:

## GOALS!! 
## 1. GET STOCK DATA IN CSV FORM.
## 2. PREPARE THE DATASET.
## 3. CREATE A MODEL.
## 4. TEST THE MODEL.
## 5. PREDICT THE FUTURE.
## 6. PLOT GRAPHS AND WHATNOT.
## 7. SAVE THE PREDICTIONS AS GRAPHS and CSVs.
## 8. Profit.

tickers = ['MMM', 'ABBV', 'ACN', 'ATVI', 'AYI', 'ADBE', 'AMD', 'AAP', 'AES', 'AET', 'AMG', 'AFL', 'A', 'APD', 'AKAM', 'ALK', 'ALB', 'ARE', 'ALXN', 'ALGN', 'ALLE', 'AGN', 'ADS', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN', 'AEE', 'AAL', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AMP', 'ABC', 'AME', 'AMGN', 'APH', 'APC', 'ADI', 'ANDV', 'ANSS', 'ANTM', 'AON', 'AOS', 'APA', 'AIV', 'AAPL', 'AMAT', 'APTV', 'ADM', 'ARNC', 'AJG', 'AIZ', 'T', 'ADSK', 'ADP', 'AZO', 'AVB', 'AVY', 'BHGE', 'BLL', 'BAC', 'BK', 'BAX', 'BBT', 'BDX', 'BRK.B', 'BBY', 'BIIB', 'BLK', 'HRB', 'BA', 'BWA', 'BXP', 'BSX', 'BHF', 'BMY', 'AVGO', 'CHRW', 'CA', 'COG', 'CDNS', 'CPB', 'COF', 'CAH', 'CBOE', 'KMX', 'CCL', 'CAT', 'CBG', 'CBS', 'CELG', 'CNC', 'CNP', 'CTL', 'CERN', 'CF', 'SCHW', 'CHTR', 'CHK', 'CVX', 'CMG', 'CB', 'CHD', 'CI', 'XEC', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG', 'CTXS', 'CLX', 'CME', 'CMS', 'KO', 'CTSH', 'CL', 'CMCSA', 'CMA', 'CAG', 'CXO', 'COP', 'ED', 'STZ', 'COO', 'GLW', 'COST', 'COTY', 'CCI', 'CSRA', 'CSX', 'CMI', 'CVS', 'DHI', 'DHR', 'DRI', 'DVA', 'DE', 'DAL', 'XRAY', 'DVN', 'DLR', 'DFS', 'DISCA', 'DISCK', 'DISH', 'DG', 'DLTR', 'D', 'DOV', 'DWDP', 'DPS', 'DTE', 'DRE', 'DUK', 'DXC', 'ETFC', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'EMR', 'ETR', 'EVHC', 'EOG', 'EQT', 'EFX', 'EQIX', 'EQR', 'ESS', 'EL', 'ES', 'RE', 'EXC', 'EXPE', 'EXPD', 'ESRX', 'EXR', 'XOM', 'FFIV', 'FB', 'FAST', 'FRT', 'FDX', 'FIS', 'FITB', 'FE', 'FISV', 'FLIR', 'FLS', 'FLR', 'FMC', 'FL', 'F', 'FTV', 'FBHS', 'BEN', 'FCX', 'GPS', 'GRMN', 'IT', 'GD', 'GE', 'GGP', 'GIS', 'GM', 'GPC', 'GILD', 'GPN', 'GS', 'GT', 'GWW', 'HAL', 'HBI', 'HOG', 'HRS', 'HIG', 'HAS', 'HCA', 'HCP', 'HP', 'HSIC', 'HSY', 'HES', 'HPE', 'HLT', 'HOLX', 'HD', 'HON', 'HRL', 'HST', 'HPQ', 'HUM', 'HBAN', 'HII', 'IDXX', 'INFO', 'ITW', 'ILMN', 'IR', 'INTC', 'ICE', 'IBM', 'INCY', 'IP', 'IPG', 'IFF', 'INTU', 'ISRG', 'IVZ', 'IQV', 'IRM', 'JEC', 'JBHT', 'SJM', 'JNJ', 'JCI', 'JPM', 'JNPR', 'KSU', 'K', 'KEY', 'KMB', 'KIM', 'KMI', 'KLAC', 'KSS', 'KHC', 'KR', 'LB', 'LLL', 'LH', 'LRCX', 'LEG', 'LEN', 'LUK', 'LLY', 'LNC', 'LKQ', 'LMT', 'L', 'LOW', 'LYB', 'MTB', 'MAC', 'M', 'MRO', 'MPC', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MAT', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK', 'MET', 'MTD', 'MGM', 'KORS', 'MCHP', 'MU', 'MSFT', 'MAA', 'MHK', 'TAP', 'MDLZ', 'MON', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MYL', 'NDAQ', 'NOV', 'NAVI', 'NTAP', 'NFLX', 'NWL', 'NFX', 'NEM', 'NWSA', 'NWS', 'NEE', 'NLSN', 'NKE', 'NI', 'NBL', 'JWN', 'NSC', 'NTRS', 'NOC', 'NCLH', 'NRG', 'NUE', 'NVDA', 'ORLY', 'OXY', 'OMC', 'OKE', 'ORCL', 'PCAR', 'PKG', 'PH', 'PDCO', 'PAYX', 'PYPL', 'PNR', 'PBCT', 'PEP', 'PKI', 'PRGO', 'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PXD', 'PNC', 'RL', 'PPG', 'PPL', 'PX', 'PCLN', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PSA', 'PHM', 'PVH', 'QRVO', 'PWR', 'QCOM', 'DGX', 'RRC', 'RJF', 'RTN', 'O', 'RHT', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RHI', 'ROK', 'COL', 'ROP', 'ROST', 'RCL', 'CRM', 'SBAC', 'SCG', 'SLB', 'SNI', 'STX', 'SEE', 'SRE', 'SHW', 'SIG', 'SPG', 'SWKS', 'SLG', 'SNA', 'SO', 'LUV', 'SPGI', 'SWK', 'SBUX', 'STT', 'SRCL', 'SYK', 'STI', 'SYMC', 'SYF', 'SNPS', 'SYY', 'TROW', 'TPR', 'TGT', 'TEL', 'FTI', 'TXN', 'TXT', 'TMO', 'TIF', 'TWX', 'TJX', 'TMK', 'TSS', 'TSCO', 'TDG', 'TRV', 'TRIP', 'FOXA', 'FOX', 'TSN', 'UDR', 'ULTA', 'USB', 'UAA', 'UA', 'UNP', 'UAL', 'UNH', 'UPS', 'URI', 'UTX', 'UHS', 'UNM', 'VFC', 'VLO', 'VAR', 'VTR', 'VRSN', 'VRSK', 'VZ', 'VRTX', 'VIAB', 'V', 'VNO', 'VMC', 'WMT', 'WBA', 'DIS', 'WM', 'WAT', 'WEC', 'WFC', 'HCN', 'WDC', 'WU', 'WRK', 'WY', 'WHR', 'WMB', 'WLTW', 'WYN', 'WYNN', 'XEL', 'XRX', 'XLNX', 'XL', 'XYL', 'YUM', 'ZBH', 'ZION', 'ZTS']
lentickers = len(tickers)

import matplotlib
import matplotlib.pyplot as plt
import datetime
import random
# plt.ion()
# matplotlib.use('Agg')
# matplotlib.interactive(True)
import time
import os

import pandas as pd

from pandas import Series

import requests
from bs4 import BeautifulSoup
import datetime
# import pandas_datareader.data as web

import matplotlib.pyplot as plt 
# rcParams['figure.figsize'] = 10,6
# plt.rcParams["figure.figsize"] = (16,9)
from stocker import Stocker
# import requests
# import time
# import os

today = datetime.datetime.today()
folder = 'predictions/'
stocksfolder = 'stocks/'
reportsfolder = 'reports/'
apikey = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX' # Add ALPHAVANTAGE API Key here.


def GetTickerSheet(ticker,keepoldfiles=False):
	url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&outputsize=full&symbol=" + ticker + "&apikey=" + apikey + "&datatype=csv"
	print(ticker, url)
	# url = str("https://api.tiingo.com/tiingo/daily/AAPL/prices?startDate="+ start +"&endDate=" + end)
	r = requests.get(url)	
	data = r.text
	soup = BeautifulSoup(data,"html.parser")
	x = str(soup).replace('\n', '')

	# if x == """{	"Information": "Thank you for using Alpha Vantage! Please visit https://www.alphavantage.co/premium/ if you would like to have a higher API call volume."}""":
	# 	print()
	datdir = stocksfolder + ticker + '.csv'
	if '"Information": "Thank you for using Alpha Vantage!' in x: 
		print(">>>WARNING WARNING ",x)
		time.sleep(10)
		GetTickerSheet(ticker)
		return
	else:
		# if 'timestamp' in x:
		# 	x.replace('date')
		with open(datdir,"w") as f:
			f.write(str(x))
			print("saved file!")
			f.close()

def GetStockPrediction(csvfile,ticker,inifile,litemode=False):
	### MAKE HISTORY GRAPH.
	hstg = 30 # number of days 

	# Define files and filetypes
	ext = ".pdf"
	ModZ =[str(stocksfolder+ticker+".HighLowClose" + ext), #0
				str(stocksfolder+ticker+".ChangeVolume"+ext),   #1
				str(stocksfolder+ticker+".ChangePoint"+ext),    #2
				# str(stocksfolder+ticker+".ChangePointTrends"+ext),#3
				str(stocksfolder+ticker+".evaluate_prediction"+ext),#4
				str(stocksfolder+ticker+".TrendsA"+ext),#5
				str(stocksfolder+ticker+".Naive" + str(hstg) + ext)]
				#6
	StockFiles=[str(stocksfolder+ticker+".HighLowClose" + ext), #0
				str(stocksfolder+ticker+".ChangeVolume"+ext),   #1
				str(stocksfolder+ticker+".ChangePoint"+ext),    #2
				str(stocksfolder+ticker+".ChangePointTrends"+ext),#3
				str(stocksfolder+ticker+".evaluate_prediction"+ext),#4
				str(stocksfolder+ticker+".TrendsA"+ext),#5
				str(stocksfolder+ticker+".Naive" + str(hstg) + ext),#6
				# ACTUAL MACHINE LEARNING BEGINS
				str(stocksfolder+ticker+".linear_regression_01"+ext)]#7

	# print(csvfile)

	# df = pd.read_csv(csvfile)
	


	## START ANALSYS.
	
	Months = str((today - datetime.timedelta(days=hstg)).date())
	### Stocker is initialized and will retrieve the entire stock's data as it is read in the file.
	stock = Stocker(ticker=ticker, 
		exchange='CSV',
		csv_repository='stocks')
	print(stock.stock.head())
	plt.clf()

	### plot_stock will plot the stock's (High, Low, Close) recent history. see 'hstg' for number of days

	stock.plot_stock(start_date = str(Months),
		stats = ['high', 'low', 'close'],
		plot_type='hlc',
		filename=StockFiles[0])
	plt.clf()

	### plot_stock will plot the Stock's (Daily Change, Volume) recent history. see 'hstg' for number of days

	stock.plot_stock(start_date = str(Months) ,
		stats = ['Daily Change', 'volume'], 
		plot_type='pct',
		filename=StockFiles[1])
	plt.clf()

	stock.changepoint_date_analysis(filename=StockFiles[2])
	plt.clf()

	# stock.buy_and_hold(start_date=str(Months),nshares=1,filename=str(stocksfolder+ticker+".PredictedBuyHold"+ext)) # This function is broken.
	# plt.clf()

	stock.evaluate_prediction(start_date=str(Months),nshares=1,filename=StockFiles[4])
	plt.clf()

	## START BASIC PREDICTIONS BASED ON ANALYSIS
	model, model_data = stock.create_prophet_model()
	model.plot_components(model_data)
	plt.savefig(StockFiles[5])
	plt.clf()

	stock.predict_future(days=hstg,filename=StockFiles[6])
	plt.clf()

	# START MORE ADVANCED PREDICTONS BASED ON DEEP LEARNING.
	if litemode:
		pass
	else:
		### changepoint_date_analysis is looking through historical data of the past 3 years to find the optimal time to buy or sell a stock.
		print(inifile)
		search_terms = inifile["STOCK PROFILE"]["search_terms"]
		# print(search_terms)
		bountylist = search_terms.split(",")
		# print(bountylist)
		stock.changepoint_date_analysis(search=bountylist,filename=StockFiles[3])
		plt.clf()
		# LINEAR REGRESSION 01
		stock.Stocker_linear_regression(days=hstg, filename=StockFiles[7])
		plt.clf()

	## Merge files into a pdf.
	
	if litemode:
		ModZ = ModZ; print(ModZ)
	else:
		ModZ = StockFiles[:]; print(ModZ)
		# thelimit = len(ModZ)
	
	Merge_PDFS(ModZ,reportsfolder+ticker+"._REPORT.PDF")
		# x = PdfFileReader(filename)


def Merge_PDFS(filenames,filename):

	from PyPDF2 import PdfFileReader, PdfFileWriter
	pdf_writer = PdfFileWriter()
	for f in filenames:
		pdf_reader = PdfFileReader(f)
		for page in range(pdf_reader.getNumPages()):
			pdf_writer.addPage(pdf_reader.getPage(page))
		with open(filename, 'wb') as fh:
			pdf_writer.write(fh)
			fh.close()

# def Merge_PDFs(files, filename="report.pdf"):
# 	print("h")

def GET_ALL_STOCKS(keepoldfiles=True):
	i = 0
	if keepoldfiles:
		for ticker in tickers:
			csvfile = (stocksfolder+ticker+".csv")
			if os.path.isfile(csvfile):
				print("SKIP",ticker)
				continue		
			else:
				GetTickerSheet(ticker)
				time.sleep(5)
				# GetStockPrediction(csvfile,ticker)
			i += 1
			print(i,len(tickers),ticker)
	else:
		for ticker in tickers:
			GetTickerSheet(ticker)
			time.sleep(5)
			i += 1
			print(i,len(tickers),ticker)
		# print(i,len(tickers),ticker)

def ParseProfile(thefilename):
	import configparser
	config = configparser.ConfigParser()
	config.read(thefilename)
	return config










def verifyMutualTickers(tickerlist):
	for ticker in tickerlist:
		if ticker in tickers:
			continue
		else:
			return False
	return True

ans=True
while ans:
	print ("""
 /$$   /$$     /$ /$     /$$                         /$$$$$$$$ /$$                                                  
| $$  /$$/              | $$                        | $$_____/|__/                                                  
| $$ /$$/     /$$$$$$   | $$$$$$$    /$$   /$$      | $$       /$$ /$$$$$$$   /$$$$$$  /$$$$$$$   /$$$$$$$  /$$$$$$ 
| $$$$$/     /$$__   $$ | $$__  $$  | $$  | $$      | $$$$$   | $$| $$__  $$ |____  $$| $$__  $$ /$$_____/ /$$__  $$
| $$  $$    | $$  \  $$ | $$  \  $$ | $$  | $$      | $$__/   | $$| $$  \ $$  /$$$$$$$| $$  \ $$| $$      | $$$$$$$$
| $$\  $$   | $$   | $$ | $$  | $$  | $$  | $$      | $$      | $$| $$  | $$ /$$__  $$| $$  | $$| $$      | $$_____/
| $$ \  $$  |   $$$$$$/ | $$$$$$$/  |  $$$$$$$      | $$      | $$| $$  | $$|  $$$$$$$| $$  | $$|  $$$$$$$|  $$$$$$$
|__/  \__/   \______/   |_______/    \____  $$      |__/      |__/|__/  |__/ \_______/|__/  |__/ \_______/ \_______/
                                    /$$  | $$                                                                      
                                   | $$$$$$/                                                                      
                                   \______/                                                                       


	1. Make Predictions about one stock.
	2. Make Predictions about one stock. (Force Download CSVs)
	3. Make Predictions about all the stocks.
	4. Make Predictions about all the stocks. (Force Download CSVs)
	5. Make Predictions about a list of stocks. [Mutual Fund Simulator] (Choose Tickers)
	6. Make Predictions about a list of stocks. [Mutual Fund Simulator] (Random Tickers)
	7. (Force Download all CSVs)


	Q. Quit
	""")
	ans=input("What would you like to do? ") 
	if ans=="1": 
		print("\n Predicting one stock.")
		ticker = input("Type in a Ticker: e.g. 'AAPL'.  >>>")
		print(ticker)
		profile = ParseProfile(stocksfolder+ticker+".ini")
		GetStockPrediction(stocksfolder+ticker+".csv",ticker,profile)
	elif ans=="2":
		print("\n Predicting one stock.")
		ticker = input("Type in a Ticker: e.g. 'AAPL'.  >>>")
		print(ticker)
		GetTickerSheet(ticker, keepoldfiles=False)
		profile = ParseProfile(stocksfolder+ticker+".ini")
		GetStockPrediction(stocksfolder+ticker+".csv",ticker,profile)
	elif ans=="3":
		print("\n Predicting all stocks. Please wait.")
		for ticker in stocks:
			print(ticker)
			profile = ParseProfile(stocksfolder+ticker+".ini")
			GetStockPrediction(stocksfolder+ticker+".csv",ticker,profile)
	elif ans=="4":
		print("\n Predicting all stocks. Please wait.")
		for ticker in stocks:
			print(ticker)
			GetTickerSheet(ticker, keepoldfiles=False)
			profile = ParseProfile(stocksfolder+ticker+".ini")
			GetStockPrediction(stocksfolder+ticker+".csv",ticker,profile)
	elif ans=="5":
		print("\n Virtual Mutual Fund. Please Wait.")
		Mutual_tickers = input("Type in ticker seperated by single spaces. i.e. 'AAPL MMM AMZN FB GOOG'.  >>>").split(" ")
		if not verifyMutualTickers(Mutual_tickers):
			print("One of your ticker names is incorrect. Please make sure you spelled it correctly and try again.")
			break
		print(Mutual_tickers)
		filebounty = []
		for ticker in Mutual_tickers:
			print(ticker)
			profile = ParseProfile(stocksfolder+ticker+".ini")
			GetStockPrediction(stocksfolder+ticker+".csv",ticker,profile,litemode=True)
			filebounty.append(reportsfolder+ticker+"._REPORT.PDF")
		x = input("Where would you like to save this completed report? >>>")
		Merge_PDFS(filebounty,x)
	elif ans=="6":
		print("\n Virtual Mutual Fund. Please Wait.")
		i = input("Enter the number of stocks you are willing to add to this portfolio. e.g. '25' '110'.\n  [NOTE] If left blank. This number will be random between 25-90. \n>>>")
		if i != '':
			i = int(i)
			if i <= 0:
				print("You entered a negative number or zero. Don't do that.")
				break
		else:
			i = random.randint(25,90)
		Mutual_tickers = random.sample(tickers, i)
		if not verifyMutualTickers(Mutual_tickers):
			print("One of your ticker names is incorrect. Please make sure you spelled it correctly and try again.")
			break
		print(i,Mutual_tickers)
		filebounty = []
		for ticker in Mutual_tickers:
			print(ticker)
			profile = ParseProfile(stocksfolder+ticker+".ini")
			GetStockPrediction(stocksfolder+ticker+".csv",ticker,profile,litemode=True)
			filebounty.append(reportsfolder+ticker+"._REPORT.PDF")
		x = input("Where would you like to save this completed report? >>>")
		Merge_PDFS(filebounty,x)
	elif ans=="Q" or "q":
		print("\n Goodbye")
		ans=False
	elif ans !="":
		print("\n Sorry, that is not a valid choice. Try again") 