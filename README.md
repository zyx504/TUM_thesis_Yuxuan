# TUM_thesis_Yuxuan
appendix
2012 data as an example
after running the code:

D:\zyxthesis\py\venv\Scripts\python.exe D:\zyxthesis\py\code.py 
Performing 5-fold cross-validation...
R2 - Cross-validation scores: [0.80386345 0.81740002 0.79006631 0.80642475 0.81167972]
R2 - Mean: 0.8059 (±0.0092)
RMSE - Cross-validation scores: [0.0651122  0.06617628 0.06547813 0.06088354 0.06232001]
RMSE - Mean: 0.0640 (±0.0020)
MAE - Cross-validation scores: [0.04689781 0.04946066 0.04936498 0.04640351 0.04780272]
MAE - Mean: 0.0480 (±0.0012)

Training final model on full training set...

Test Set Performance:
R2: 0.8584
RMSE: 0.0574
MAE: 0.0429
 SHAP Value range: -0.13723616 ~ 0.45461729

📌 Feature: total_pop
  Feature value range: 85.00000000 ~ 9840024.00000000
  Bin 1: 85.000000 - 1855.302000 
  Bin 2: 1855.302000 - 7629.164000 
  Bin 3: 7629.164000 - 25880.000000 
  Bin 4: 25880.000000 - 118391.194000 
  Bin 5: 118391.194000 - 744906.390000 
  Bin 6: 744906.390000 - 9840024.000000 

📌 Feature: sex_ratio
  Feature value range: 73.40000000 ~ 325.00000000
  Bin 1: 73.400000 - 88.725400 
  Bin 2: 88.725400 - 94.058200 
  Bin 3: 94.058200 - 98.100000 
  Bin 4: 98.100000 - 104.600000 
  Bin 5: 104.600000 - 127.974600 
  Bin 6: 127.974600 - 325.000000 

📌 Feature: pct_black
  Feature value range: 0.00000000 ~ 0.86200000
  Bin 1: 0.000000 - 0.000000 
  Bin 2: 0.000000 - 0.006000 
  Bin 3: 0.006000 - 0.027000 
  Bin 4: 0.027000 - 0.215418 
  Bin 5: 0.215418 - 0.555000 
  Bin 6: 0.555000 - 0.862000 

📌 Feature: pct_hisp
  Feature value range: 0.00000000 ~ 0.98300000
  Bin 1: 0.000000 - 0.003000 
  Bin 2: 0.003000 - 0.012000 
  Bin 3: 0.012000 - 0.033000 
  Bin 4: 0.033000 - 0.141000 
  Bin 5: 0.141000 - 0.533492 
  Bin 6: 0.533492 - 0.983000 

📌 Feature: pct_bach
  Feature value range: 0.03700000 ~ 0.72800000
  Bin 1: 0.037000 - 0.084254 
  Bin 2: 0.084254 - 0.120000 
  Bin 3: 0.120000 - 0.172000 
  Bin 4: 0.172000 - 0.272000 
  Bin 5: 0.272000 - 0.438746 
  Bin 6: 0.438746 - 0.728000 

📌 Feature: median_income
  Feature value range: 19624.00000000 ~ 122844.00000000
  Bin 1: 19624.000000 - 27513.970000 
  Bin 2: 27513.970000 - 35209.000000 
  Bin 3: 35209.000000 - 43741.000000 
  Bin 4: 43741.000000 - 54516.688000 
  Bin 5: 54516.688000 - 77141.474000 
  Bin 6: 77141.474000 - 122844.000000 

📌 Feature: pct_65_over
  Feature value range: 0.04100000 ~ 0.44500000
  Bin 1: 0.041000 - 0.087254 
  Bin 2: 0.087254 - 0.122000 
  Bin 3: 0.122000 - 0.157000 
  Bin 4: 0.157000 - 0.201000 
  Bin 5: 0.201000 - 0.260746 
  Bin 6: 0.260746 - 0.445000 

📌 Feature: pct_age_18_29
  Feature value range: 0.03300000 ~ 0.57000000
  Bin 1: 0.033000 - 0.089000 
  Bin 2: 0.089000 - 0.115000 
  Bin 3: 0.115000 - 0.139000 
  Bin 4: 0.139000 - 0.173000 
  Bin 5: 0.173000 - 0.266746 
  Bin 6: 0.266746 - 0.570000 

📌 Feature: gini
  Feature value range: 0.33220000 ~ 0.59940000
  Bin 1: 0.332200 - 0.371651 
  Bin 2: 0.371651 - 0.401300 
  Bin 3: 0.401300 - 0.432800 
  Bin 4: 0.432800 - 0.470542 
  Bin 5: 0.470542 - 0.512198 
  Bin 6: 0.512198 - 0.599400 

📌 Feature: pct_manuf
  Feature value range: 0.00000000 ~ 0.41100000
  Bin 1: 0.000000 - 0.015000 
  Bin 2: 0.015000 - 0.051000 
  Bin 3: 0.051000 - 0.114000 
  Bin 4: 0.114000 - 0.192000 
  Bin 5: 0.192000 - 0.273746 
  Bin 6: 0.273746 - 0.411000 

📌 Feature: ln_pop_den
  Feature value range: -3.01451365 ~ 10.21130642
  Bin 1: -3.014514 - -0.709904 
  Bin 2: -0.709904 - 1.220211 
  Bin 3: 1.220211 - 2.864679 
  Bin 4: 2.864679 - 4.433481 
  Bin 5: 4.433481 - 6.536791 
  Bin 6: 6.536791 - 10.211306 

📌 Feature: pct_fb
  Feature value range: 0.00000000 ~ 0.51200000
  Bin 1: 0.000000 - 0.003000 
  Bin 2: 0.003000 - 0.009000 
  Bin 3: 0.009000 - 0.024000 
  Bin 4: 0.024000 - 0.074418 
  Bin 5: 0.074418 - 0.220746 
  Bin 6: 0.220746 - 0.512000 

📌 Feature: pct_insured
  Feature value range: 0.54800000 ~ 0.97400000
  Bin 1: 0.548000 - 0.718000 
  Bin 2: 0.718000 - 0.798000 
  Bin 3: 0.798000 - 0.855000 
  Bin 4: 0.855000 - 0.905000 
  Bin 5: 0.905000 - 0.937000 
  Bin 6: 0.937000 - 0.974000 

📌 Feature: Latitude
  Feature value range: 25.58611980 ~ 48.84265310
  Bin 1: 25.586120 - 29.347905 
  Bin 2: 29.347905 - 32.955117 
  Bin 3: 32.955117 - 38.306180 
  Bin 4: 38.306180 - 43.478578 
  Bin 5: 43.478578 - 47.577933 
  Bin 6: 47.577933 - 48.842653 

📌 Feature: Longitude
  Feature value range: -124.21092920 ~ -67.60935420
  Bin 1: -124.210929 - -120.812924 
  Bin 2: -120.812924 - -101.748922 
  Bin 3: -101.748922 - -90.213792 
  Bin 4: -90.213792 - -80.883138 
  Bin 5: -80.883138 - -73.853561 
  Bin 6: -73.853561 - -67.609354 
Exported GeoPackage, can be loaded directly in ArcGIS Pro: D:\zyxthesis\shap_output\shap_results_zzzz.gpkg
Generating mean absolute SHAP bar plot...
Feature importance bar chart saved as 'shap_mean_absolute_bar_chart.png'
Generating SHAP summary plot...
SHAP summary plot saved as 'shap_summary_plot.png'

Process finished with exit code 0
