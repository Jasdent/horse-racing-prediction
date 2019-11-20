Teamname: eisodjehg

Team member / Student id:
	CHEN, Mo	20633437
	YANG, Zhuokun   20619443

**About the backtesting result**
The workflow of this submission is down in the following:
1. configure mode = "train" in the utils_global.py, then run backtesting.py to get the results (regarding the given training file ONLY) AS WELL AS train the model with 	HR200709to201901.csv, which has its own trainand test split. 
2. configure mode = "test", and run backtesting.py; **the final "backtesting_summary_Python.csv"** (the one you are seeing in current folder) is the RMSE of the completely unseen data "Sample_test.csv".


**Python Script Description**
- utils_global.py: including all common packages and global variable that should be used in ALL other files. **This file is designated for instructor mode**:
	1. the global variable *mode* can either be set "train" or "test". By making it "train" and run backtesting.py, the pipeline would use the HR200709to201901.csv to train (i.e. if the model.pkl exists, it will simply load the model and perform evaluation on HR200709to201901.csv); by making it "test", the pipeline would simply preprocess the new data, feed it to existing model.
	2. the global variable *FILEPATH* under test mode should be modified to new input files "[newfilename].csv". 
- model.py : containing a class with random forest model including some functionality of testing and validation
- predict.py: used to predict the *4 outputs* which are then injected to the original dataframe, please do not run this alone; use it binded with backtesting.py. The output dataframe will also be saved to "./data/results_test.csv".
- train.py: *should only be used by student* as for getting a new model and perform evaluation of seen data.
-backtesting.py: generates results from inputs. 

**NOTE for instructor**
The instructor shall modify the "FILEPATH" and "mode" (i.e. should be mode = "test") global variable in utils_global.py, then simply run backtesting.py. FILEPATH should be the new data of the current week.

It is suggested to put the new csv file in "./data/". 
