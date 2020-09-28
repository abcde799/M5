# M5
Top 13.6% solution with simple linear regression model.

The idea details are in the notebook called 'beat_baseline_take1'. The goal of the competition is to predict a 28 day sales horizon
for over 30000 individual time series corresponding to the sales of various products from various Walmarts in the USA. The metric for 
evaluation is a variant of the root mean squared error (RMSE). 

The strategy is as follows: We produce a time series forecast for each item, giving rise to 30000 individual forecasts. This approach avoids
aggregation bias, but at the same time may not capture cross correlation between different products. Nonetheless, it worked well enough and
wasn't hard on either memory or runtime constraints. The work was performed on Google Colab and the final training took just under four 
hours.  


Since we wish to minimize a variant of the RMSE, we take as a baseline the rolling mean of the past 28 days as
a constant prediction for the next 28 days, for each time series. 

To beat this baseline, a lagged linear regression model is used which incorporates this baseline 
as a feature among other features. The final model is a linear regression which includes 28-37 day lags as well as the corresponding rolling
means, medians, and rolling year mean and median. Weekdays are enocded using one-hot encoding as well as the weekends. 
Holiday information is not included, which likely would have improved the score. Finally, there is a Lasso penalty which is crucial for the 
performance of the model.

Since we are dealing with time-series, cross validation doesn't really make sense. Instead we take the most recent period of 28 days before 
the 28 day forecast horizon, and use that as a test period. We evaluate forecast performance by using RMSE as the metric, and by taking random 
samples of items and looking at the individual model performance for a set of Lasso hyperparameters, compared with the baseline. Once
the hyperparameter is chosen, by taking a large random sample of individual time series from the testing period and observing how many times 
our model beats the baseline, we can rule out the null hypothesis that the probability of beating the baseline on a randomly selected product 
is less than or equal to 50%. This approach makes sense since the 'best' estimate of the forecasting error is likely the most recent 28 day 
window, just before the 28 day forecasting horizon. 




