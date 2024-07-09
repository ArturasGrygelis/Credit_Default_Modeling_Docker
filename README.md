Embarking on a solo data-driven entrepreneurial journey, our startup is set to revolutionize risk evaluation services for retail banks. Armed with a dataset from Home Credit Group, we're on a mission to employ machine learning for predicting loan defaults in our proof-of-concept (POC) product.

As both the data scientist and product lead, my objective is to navigate through datasets, including those from Credit Bureau, to unveil pivotal patterns. The focus is on identifying and prioritizing significant features, leading to the creation of a classification model targeting credit defaults (1 for default, 0 for non-default).

This strategic endeavor involves not only technological exploration but also the creation of a Docker image with an API endpoint, hosted on a Cloud, showcasing our commitment to delivering impactful solutions that redefine risk assessment for retail banks.

Order of Notebook used in analysis and modeling:
1.Tables_preparation_EDA
2.Combined_data_EDA
3.Best_set_of_features
4.1.Models_evaluation-RFA_features
4.2.Models_evaluation_38features
Information about final model and Proof of concept is in 4.2.Models_evaluation_38features notebook

Api endpoint created with fast api is in api folder. Api is hosted on https://credit-default-prediction-x14o.onrender.com/predict. Only request Api endpoint is set up.

Best created model is Light Gradient Boosting Classifier model.
Model , with probability threshold of 0.5 indicates 63% of credits, that's defaulted.

