# LoanSafe
A web application which can do real time risk assesment to help drive lending decisions.

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




