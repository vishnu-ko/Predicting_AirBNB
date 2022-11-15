# Predicting_AirBNB_Price
---------------------------------------

#### Vishnu Kodicherla
-----------

### Executive Summary

Began in 2008 as AirBed and Breakfast, the company is now known as AirBnB. The company is notoriosuly known for its operations being online and allowing for short-term home stays and experiences. With the growing number of Host since 2008, AirBnB has been working towards being a sucessful company. With new hosts joining everyday, many have not been able to understanding the correct pricing to optimize revenue and satisfy customers. To address the concerns of Hosts and even Customers, a machine learning model was made to help with providing pricing predictions in the area of New York City. With this model, it will provide Hosts and Customers an expedited opportunity to choose and evaluate different AirBnB prices. 

The data that was collected includes 39,000 set of different AirBnBs throughout the New York City vicinity. The data was obtained from InsiderAirBnB and was provided with a number of features. Each feature was looked at and identified to see if it provides assistance in being able to predict the AirBnB price. The data was compiled on September 7, 2022. 

Using a number of different models to analyze the data, I built multiple models to see what model predicted the AirBnB with the highest R squared values. The models that were used were Linear Regression, Lasso Regularization, Ridge Regularization, Bagging Regressors, RandomForest Regressor, and then ran two Neural Networks using Keras from Tensorflow. I trained all the models to look into Regression metrics in which being able to take the features and evaluate the price based off of the features. 

Optimizing for the R squared values to be over 0.55 for the testing set of the data, after evaluating each of the models, my Linear Regression with Lasso Regularization performed the best compared to the other models. The R squared values that were obtained from this model was 0.65 for the train set and 0.65 for the test set. The scores were chosen as the best model due to not being overfit compared to the other models, for example, the two neural networks that were performed and Bagging regressor. I believe to improve the model, it would be necessary to look further in depth into the amenities column as well as focus more on the distributions of each of the features and try to make them closer to a normal distribution.

The results obtained from the model allows for hosts and customer to look into pricing for AirBnB based off of looking at the amenities that were the most common as well as what type of rental unit they are interested in leasing or booking. 

-----

### Problem Statement

With the growing number of hosts and customers for Airbnb in the past decade, I was assigned to help them in operational matters which include helping new hosts in being able to gauge the price they should be selling and help with them to establish a base price. Due to New York City having a lot of competition, it is necessary to provide a baseline understanding of the data. To be able to analyze the base price, different models were used including Linear Regression, Lasso Regularization, Ridge Regularization, Bagging Regressors, RandomForest Regressor, and then ran two Neural Networks using Keras from Tensorflow. An efficient model will be based off of the one with the highest R squared scores. 

---


### Datasets
The Data was obtained from InsiderAirBnB and focused on New York City, United States of America. Insider AirBnB looks into how AirBNB influences residential communities when renting out property to tourists. 

The Datasets included 41 features and 39,881 rows. The features that were kept for modeling are shown below :
|Feature|Type|Dataset|Description|
|---|---|---|---|
Id|int64|df_cleaning_01|ID number of AirBnB
host_response_rate|float64|df_cleaning_01|# of times a host leaves a respond 
host_acceptance_rate|float64|df_cleaning_01|# of times a host accepts an AirBnB
host_is_superhost|float64|df_cleaning_01|Indicating if the host is a superhost
host_listings_count|float64|df_cleaning_01|# of host AirBnB listings 
host_has_profile_pic|int64|df_cleaning_01|Indicating if a host has an AirBnB profile picture 
host_identity_verified|int32|df_cleaning_01|Indicating if a host is verified in AirBnB
neighbourhood_cleansed|object|df_cleaning_01|Indicating Groups of Neighborhood
latitude|float64|df_cleaning_01|The latitude coordinates of the AirBnB property
longitude|float64|df_cleaning_01|The longitude coordinates of the AirBnB property
property_type|object|df_cleaning_01|The AirBnB property type
room_type|object|df_cleaning_01|The room type of the AirBnB property
accommodates|int64|df_cleaning_01|Number of people accommodate in the AirBnB property 
bathrooms|object|df_cleaning_01|# of bathrooms of AirBnB property
bedrooms|float64|df_cleaning_01|# of bedrooms of AirBnB property
beds|float64|df_cleaning_01|# of beds of AirBnB property
price|float64|df_cleaning_01|Price per night of the AirBnB property
minimum_nights|int64|df_cleaning_01|The minimum number of nights that a customer will stay 
maximum_nights|int64|df_cleaning_01|The maximum number of nights that a customer will stay
has_availability|int32|df_cleaning_01|The availability status of the AirBnB property
availability_30|int64|df_cleaning_01|The availability to rent a AirBnB property for 30 days
availability_365|int64|df_cleaning_01|The availability to rent a AirBnB property for 365 days
number_of_reviews|int64|df_cleaning_01|# of reviews of an AirBnB property
number_of_reviews_ltm|int64|df_cleaning_01|# of reviews for the last 12 months of an AirBnB property 
number_of_reviews_l30d|int64|df_cleaning_01|# of reviews for the last 30 days of an AirBnB property 
review_scores_rating|float64|df_cleaning_01|Review score rating out of 5 for AirBnB property
review_scores_accuracy|float64|df_cleaning_01|Review score of AirBnB property in terms of accuracy in the property’s description
review_scores_cleanliness|float64|df_cleaning_01|Review score of AirBnB property in terms of accuracy in the property’s cleanliness
review_scores_checkin|float64|df_cleaning_01|Review score of AirBnB property in terms of accuracy in the property’s checkin
review_scores_communication|float64|df_cleaning_01|Review score of AirBnB property in terms of accuracy in the property’s communication
review_scores_location|float64|df_cleaning_01|Review score of AirBnB property in terms of accuracy in the property’s location
review_scores_value|float64|df_cleaning_01|Review score of AirBnB property in terms of accuracy in the property’s value
instant_bookable|int64|df_cleaning_01|Indicates if an AirBnB can be booked instantly
reviews_per_month|float64|df_cleaning_01|# of reviews that an AirBnB property receives per month 
tv|int32|df_cleaning_01|Indicates if AirBnB property has any of the following electronics including: TV, Bluetooth sound system, Xbox One, Amazon Prime Video, PS4
heating|int32|df_cleaning_01|Indicates if AirBnB property has any source of heating
gym|int32|df_cleaning_01|Indicates if AirBnB property has any available gym including: Exercise Equipment, and gym
internet|int32|df_cleaning_01|Indicates if AirBnB property has any internet including: Internet, Wifi, Fast Wifi 
parking|int32|df_cleaning_01|Indicates if AirBnB property has any of the following parking availability including: Free Street Parking, Free Parking on Premis, Paid Parking off Premise 
kitchen_supply|int32|df_cleaning_01|Indicates if AirBnB property has any of the following kitchen supplies including: Microwave, Coffee Maker, Oven, Kitchen, Stove, Dishes and Silverware, Dishwasher, Refrigerator, Cooking basics
bathroom_supply|int32|df_cleaning_01|Indicates if AirBnB property has any of the following bathroom supplies including: Hair Dryer, Shampoo, and Bathtub
laundry|int32|df_cleaning_01|Indicates if AirBnB property has any of the following laundry appliances including: Iron, Washer, Dryer 
entrance|int32|df_cleaning_01|Indicates if AirBnB property has any private entrances
stay|int32|df_cleaning_01|Indicates if AirBnB property has any long term stay availability
secure|int32|df_cleaning_01|Indicates if AirBnB property has any security systems in place including: Security Cameras on Property, Lock on bedroom doors, Keypad, and Smart Lock



### Data Cleaning/ Null values
I began by looking at each of the columns and how they influence prices of AirBnB. I checked to see if there were any null values in the columns and saw that there were some columns with a low number of null values and other columns that had a large number of null values. The columns with a low number of null values were host_is_superhost, host_listing_counts, host_has_profile_pic, host_identity_verified, and bathrooms. Each of these columns had a low number of null values that was less than 5% of the data. I then began looking at each of the columns and decided to drop any that included URLs and scraping data because they aren’t useful in providing model predictions when they are just URLs. I then dropped a lot information that was presented about hosts because that won’t provide more knowledge on how to improve the prediction of the prices of AirBnB properties. 

I then dropped multiple columns including bathrooms, calendar_updated, and license because all the rows were full of null values making them insignificant to use. I then began to look at each columns’ dtypes and made appropriate changes for each of the columns that had incorrect dtype categorization. 

Some columns were looked into further including Amenities and property types. There were a wide range of amenities that were provided by each AirBnB. Out of the amenities that were included in the original dataset, 12 amenities were ultimately kept which are detailed below. When it came to property types, there were different types of rental properties. To shorten it out and make it concise, I made 4 categories which includes: Entire Rental Unit, Private Room Rental Unit, Shared Room rentals units, and Other. 

|Feature|Type|Dataset|Description|
|---|---|---|---|
tv|int32|df_cleaning_01|Indicates if AirBnB property has any of the following electronics including: TV, Bluetooth sound system, Xbox One, Amazon Prime Video, PS4
heating|int32|df_cleaning_01|Indicates if AirBnB property has any source of heating
gym|int32|df_cleaning_01|Indicates if AirBnB property has any available gym including: Exercise Equipment, and gym
internet|int32|df_cleaning_01|Indicates if AirBnB property has any internet including: Internet, Wifi, Fast Wifi 
parking|int32|df_cleaning_01|Indicates if AirBnB property has any of the following parking availability including: Free Street Parking, Free Parking on Premis, Paid Parking off Premise 
kitchen_supply|int32|df_cleaning_01|Indicates if AirBnB property has any of the following kitchen supplies including: Microwave, Coffee Maker, Oven, Kitchen, Stove, Dishes and Silverware, Dishwasher, Refrigerator, Cooking basics
bathroom_supply|int32|df_cleaning_01|Indicates if AirBnB property has any of the following bathroom supplies including: Hair Dryer, Shampoo, and Bathtub
laundry|int32|df_cleaning_01|Indicates if AirBnB property has any of the following laundry appliances including: Iron, Washer, Dryer 
entrance|int32|df_cleaning_01|Indicates if AirBnB property has any private entrances
stay|int32|df_cleaning_01|Indicates if AirBnB property has any long term stay availability
secure|int32|df_cleaning_01|Indicates if AirBnB property has any security systems in place including: Security Cameras on Property, Lock on bedroom doors, Keypad, and Smart Lock

---
### Preprocessing Models
For the preprocessing section, three main events occurred which were Dummifying the columns, using an Iterative Imputer, and checking variance inflation factor. 

From the Data Dictionary that was presented above, the columns that were needed to be dummified were neighbourhood cleansed, property type, room type, and bathrooms. Each of these columns were categorical and to be able to include them into the modeling section, they needed to be changed. Each group was looked at to make sure there were no null values present before the model phase as well.

For the iterative imputer, some of the columns had a large number of null values that exceeded 5% of the data. The columns include host_response_rate, host_acceptance_rate, bedrooms, beds, review_score_ratings, review_score_accuracy, review_score_cleanliness, review_scores_checkin, review_scores_communication, review_scores_location, review_scores_values, and reviews_per_month. A linear regression imputer was used for the null values for the train and testing datasets. 

The last preprocessing used was VIF also known as variance inflation factor. This looked into multicolinearity between columns. A score that is greater than 5 informs that two independent features have a high intercorrelation between them. When it was conducted, it was found that host_response_rate and host_acceptance_rate had very high intercorrelation, as a result, I began by dropping the higher VIF number which was host_acceptance_rate from the model to hinder the high intercorrelation. 



### Models
Multiple models were looked at including Linear Regression, Lasso Regularization, Ridge Regularization, Bagging Regressors, Random Forest Regressor, and two Deep learning Neural Networks. The models were made to predict prices for AirBnB properties with the features that were chosen after data cleaning. 

When looking at the Linear Regression R squared values, the training score was 0.68, but the testing score was -9.0265. This is after applying the natural log of the price column since the distribution was left skewed where I attempted to normalize the distribution a little more. The model was too overfit and so the testing set did a really bad job even after applying standard scalar for the x_train and x_test values. 

After conducting the Linear Regression, Lasso was conducted in which the natural log of y_train and y_test were used to normalize the distribution. The R squared value for the training score was 0.65 and the testing score was 0.65. The best parameter for Lasso had an alpha of 0.01 with a best score of 0.63. This model was not overfit and did the best compared to the other models. 

After conducting Linear Regression and Lasso, ridge regularization in which the natural log of the y_train and y_test values were used to normalize the distribution. The R squared value for the training score was 0.62 and the testing score was 0.63. The best parameter for Ridge had an alpha of 17475.284 and a best score of 0.517. When comparing to the other models, this came to be the second best model.

A bagging Regressor was done in which there was an observation of overfitting that occurred. The R squared value for the training set was 0.86 and the testing set was 0.44. The parameters for the bagging regressor had a max features value of 0.85, max samples value of 0.8, and n_estimators values of 15. 

The Random Forest also provided scores that were overfit as well.  Its R squared value for the training set was 0.72 and the testing set was 0.45. The parameters for Random Forest selected had a max_depth value of 15, max_features of sqrt, max_samples values of 0.95, and n_estimator values of 200. 

For the last two models, a Neural Network was performed on the data. From this Neural Network, 100 epochs were run with a batch_size of 256. The train set R squared value was 0.64 and the testing set was 0.58. There was a small overfitting, but overall the model performed pretty well and similar to the Regularization models.





### Conclusions
When looking at next steps, I would focus more into the data and see how logging the features that had skewed distributions would affect the models. I would also make more neural networks with more dense layers to see how that would influence the models as well. I also felt that NLP modeling with predicting the models would help as well when it comes to accurately predicting the Airbnb price. That will be the next plan after this. 


### References
http://insideairbnb.com/about/
https://en.wikipedia.org/wiki/Airbnb




