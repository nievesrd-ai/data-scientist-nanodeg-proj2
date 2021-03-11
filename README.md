## Project Overview and Motivation
This project is requirement of Udacity's Data Scientist nanodegree program. It leverages data provided by figure eight. The data has a number of tweeter messages related to a natural disaster. The project applies
ELT pipelines, trains a ML model and interfaces a web front end that assigns category labels to a sample tweet message. the web page includes some informational graphics as well.

## Project Instructions

1. Clone the repository and navigate to the downloaded folder.
	
	```	
		git clone https://github.com/nievesrd-ai/data-scientist-nanodeg-proj2.git
		cd data-scientist-nanodeg-proj2/app
	```
2. This project was built and tested for python 3.9. Make sure you have already installed the necessary Python packages according to `requirements.txt`

3. Run the following commands in the project's root directory to set up your database and model.
	- To run ETL pipeline that cleans data and stores in database
		`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
	- To run ML pipeline that trains classifier and saves
		`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
	- To prepare your enviroment tu run the web app
		`pip install plotly -U`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

5. Go to http://0.0.0.0:3001/


## Built With

* [Pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html#min)
* [NumPy](https://numpy.org/doc/stable/contents.html)
* [scikit-learn](https://scikit-learn.org/stable/)
* [plotly](https://plotly.com/python/)

## Files in Repo
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md
- LICENSE Legal information about license
- requirements.txt - list of packages needed to run the app

## Summary of Results

* The model predicts Category of an input message with an accuracy of 94%


## License

Distributed under the MIT License. See `LICENSE` file for more information.

## Project Links

* [Git](https://github.com/nievesrd-ai/data-scientist-nanodeg-proj2.git)

## Acknowledgements

* [StackOverFlow](https://stackoverflow.com/)