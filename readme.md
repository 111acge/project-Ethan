```
Load_2025
│   BNN_predictor.py
│   CNN_predictor.py
│   data_preprocessor.py
│   LSTM_predictor.py
│   main.py
│   MLP_predictor.py
│   ModelEvaluator.py
│   prophet_predictor.py
│   RandomForests_ensemble.py
│   readme.md
│   Stack_ensemble.py
│   style.py
│   time_series_predictor.py
│   visualizer.py
│   XGBoost_ensemble.py
├───data
│   ├───Australia
│   │   │   a.zip
│   │   │   b.zip
│   │   │   c.zip
│   │   │   d.zip
│   │   │   
│   │   ├───a
│   │   │       forecastdemand_vic.csv
│   │   │       
│   │   ├───b
│   │   │       forecastdemand_sa.csv
│   │   │       
│   │   ├───c
│   │   │       forecastdemand_qld.csv
│   │   │       
│   │   └───d
│   │           temprature_qld.csv
│   │           temprature_sa.csv
│   │           temprature_vic.csv
│   │           totaldemand_qld.csv
│   │           totaldemand_sa.csv
│   │           totaldemand_vic.csv
│   │           
│   └───NSW
│           forecastdemand_nsw.csv.zip.partaa
│           forecastdemand_nsw.csv.zip.partab
│           temperature_nsw.csv
│           temperature_nsw.csv.zip
│           totaldemand_nsw.csv
│           totaldemand_nsw.csv.zip
│
├───plots 
├───logs
├───RMSE_results
└───model_outputs
        

```

---


temperature_nsw.csv
```CSV
LOCATION,DATETIME,TEMPERATURE
Bankstown,1/1/2010 0:00,23.1
Bankstown,1/1/2010 0:01,23.1
Bankstown,1/1/2010 0:30,22.9
Bankstown,1/1/2010 0:50,22.7
Bankstown,1/1/2010 1:00,22.6
```

totaldemand_nsw.csv0
```CSV
DATETIME,TOTALDEMAND,REGIONID
1/1/2010 0:00,8038,NSW1
1/1/2010 0:30,7809.31,NSW1
1/1/2010 1:00,7483.69,NSW1
1/1/2010 1:30,7117.23,NSW1
1/1/2010 2:00,6812.03,NSW1
```

---

```commandline
pip3 install -r requirements.txt
```