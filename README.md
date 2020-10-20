# LoanSafe
A web application which can do real time risk assesment to help drive lending decisions.


## Companies collaborating with Insight, please make sure you use below URL to access the web application(The link on packets you recieved may be obsolete.)
## Live application URL: http://35.226.165.34:8080/

# Motivation
LoanSafe is an application designed to help financial institutions conduct risk assesment on applicants who lack a credit history. It is streamlit web application
contrainerized through Docker and deployed to a Kubernetes cluster through Google Cloud Platform and can make risk assesment in real time.LoanSafe provides a choice
to user to choose a array of applicant features and also select from five machine learning models to run the risk assesment:

1. Logistic Regression
2. Random Forrest
3. XGBoost 
4. Neural Network
5. Deep Neural Networkm with Pretrained embeddings

# Data Pipeline

![Pipeline](/images/data_pipeline.png)

# Tools/Packages

You can simply install the following into your virtual enviroment using pip.

``` pip install tensorflow ```


1. Keras
2. Tensorflow 2
3. Python 3.7
4. Pandas
5. streamlit
6. XGBoost
7. Scikit Learn 

## Containerzation
Docker is the recommend tool to have this streamlit application up and running on your local system. You can download the install from thier official website and follow the instructions there to have Docker running on your Desktop/Mac/Linux. You simply need to have a "requirments.txt" and a "Dockerfile"(no extension) in your source folder, and you can use the following two commands to build and run yor Docker image on localhost.

```
docker build -t image_name:image_version -f Dockerfile . 
```

```
docker run -p source_port:destination_port image_name:image_version 
```


## Approach

I trained embeddings for the categorical variables through a non deep neural network and saved the embeddings in a .csv file which were later feeded to deep neural network to obtain better results. This embeddings could potentially be used to compare categorical variable through a metric like cosine similarity which otherwise wouldnt be possible.The "max length" defined for each categorical variable was 1 where the vocabulary size depended on number of unique values that a particular categorical variable can assume.

## Comparision Metric

To evaluate the performances of each model I have used Recall scores and below are the results obtained.The reason I chose this metric is because I looked at this problem from the point of view of the financial institutions
and maximising the recall score would enable them to identify applicants who default to a great extent which would help these institutions greatly.Some other possible route to take would be to work on precision or f1 scores,
it essentially would all depend on who your potential users will be and what they want most from this application.

XGBoost performed the best in terms of recall scores with Neural Networks in close second and way ahead of models like random forrest and logistic regression.One reason for boosting approach(XGBoost) 
greatly outperforming the bagging approach(Random Forrest) was due to the fact like bagging approaches tend to do well when there is a clear case of overfitting happening with your models and having numerous 
sub sampled datasets helps greatly in reducing the variance and in effect the overfitting but as this was not the case here with the ML models and rather it was a case of underfitting(generally speaking)
the boosting approach fared very well.

![Scores](/images/model_scores.png)

## Refrences

1. Word embeddings. (n.d.). Retrieved October 20, 2020, from Tensorflow.org website: https://www.tensorflow.org/tutorials/text/word_embeddings

2. Tzoufras, M. (n.d.). LendingAtlas.

3 .Faircloth, B. (2019, November 6). The risks and management of algorithmic bias in fair lending. Retrieved October 20, 2020, from Wolterskluwer.com website: https://www.wolterskluwer.com/en/expert-insights/the-risks-and-management-of-algorithmic-bias-in-fair-lending

4. Klein, A. (2020, July 10). Reducing bias in AI-based financial services. Retrieved October 20, 2020, from Brookings website: https://www.brookings.edu/research/reducing-bias-in-ai-based-financial-services/

