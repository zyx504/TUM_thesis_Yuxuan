# TUM_thesis_Yuxuan
appendix

each year include:

1 "shap_mean_absolute_bar_chart.png": 
Display the mean absolute SHAP values for each feature, providing a global overview of predictor importance across all counties.

1 "shap_summary_plot.png": 
Offer detailed insights into feature effects through a multivariate display where each point represents a county observation. The horizontal position indicates the SHAP value's magnitude and direction, while point coloring represents the actual feature value from low (blue) to high (red).

15 individual feature SHAP maps: 
Color intensity corresponds to the magnitude of SHAP values,where blue hues represent negative contributions (reducing Democratic vote share) and red hues represent positive contributions (increasing Democratic vote share). 
Point sizing was implemented using a quantile-based classification with six size categories to represent the distribution of actual feature values, where larger points indicate higher feature values.
