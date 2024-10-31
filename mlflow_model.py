import mlflow 
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt 

if __name__== "__main__":
    experiment = mlflow.create_experiment(
        'testing_models_log_models',
        #artifact_location = "testing_mlflow1_artifacts",
        tags = {"env":"dev","version":"1.0.0"},
    )
    
    mlflow.set_experiment(experiment_name="testing_models_log_models")
    with mlflow.start_run(run_name="logging_models_lib")  as run:
        
        X,y = make_classification(n_samples=1000, n_features=10, n_informative=5,n_redundant=5,random_state=42)
        X_train, X_test,y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=43)
        
    
        #log model
        mlflow.autolog()
        
        
        rfc = RandomForestClassifier(n_estimators=100, random_state=42)
        rfc.fit(X_train,y_train)
        y_pred = rfc.predict(X_test)
        
        ###logmodel (same as auto log)
        mlflow.sklearn.log_model(sk_model = rfc, artifact_path="random_forest_classifier")
        #mlflow.sklearn.autolog()
        
        
        #print run info 
        print("run-id:{}".format(run.info.run_id))
        print("experiment_id:{}".format(run.info.experiment_id))
        print("status:{}".format(run.info.status))
        print("start_time:{}".format(run.info.start_time))
        print("end_time:{}".format(run.info.end_time))
        print("lifecycle_stage:{}".format(run.info.lifecycle_stage))