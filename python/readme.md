# python

The python file 'buildTestJSON.py' is used to create an input file for the integration test, which is performed by this library.
As input for the integration test the file 'marketingData.csv', which resides in the 'test/data' folder, is used. It is obtained from the kaggle repository https://www.kaggle.com/fayejavad/marketing-linear-multiple-regression
'buildTestJSON.py' uses the file to compute a multiple linear regression and to write the results to a json file called 'marketingPythonResult.json' in the 'test/data' folder.
