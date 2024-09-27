import mlflow 
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt 

if __name__== "__main__":
    experiment = mlflow.create_experiment(
        'testing_images',
        #artifact_location = "testing_mlflow1_artifacts",
        tags = {"env":"dev","version":"1.0.0"},
    )
    
    mlflow.set_experiment(experiment_name="testing_images")
    with mlflow.start_run(run_name="logging_images")  as run:
        
        X,y = make_classification(n_samples=1000, n_features=10, n_informative=5,n_redundant=5,random_state=42)
        X_train, X_test,y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=43)
        
        
        rfc = RandomForestClassifier(n_estimators=100, random_state=42)
        rfc.fit(X_train,y_train)
        y_pred = rfc.predict(X_test)
        
        
        ##log the precision-recall curve
        
        fig_pr =plt.figure()
        pr_display = PrecisionRecallDisplay.from_predictions(y_test,y_pred,ax=plt.gca())
        plt.title("Precision-Recall Curve")
        plt.legend()
    
        mlflow.log_figure(fig_pr,"images/precision_recall_curve.png")
    
        ##log the ROC curve
        fig_roc =plt.figure()
        roc_display = RocCurveDisplay.from_predictions(y_test,y_pred,ax=plt.gca())
        plt.title("ROC curve")
        plt.legend()
    
        mlflow.log_figure(fig_roc,"images/roc_curve.png")
        
        ##log the confusion matric
        fig_cm =plt.figure()
        cm_display = ConfusionMatrixDisplay.from_predictions(y_test,y_pred,ax=plt.gca())
        plt.title("Confusion Matrix")
        plt.legend()
    
        mlflow.log_figure(fig_cm,"images/confusion_matrix.png")
        
        #print run info 
        print("run-id:{}".format(run.info.run_id))
        print("experiment_id:{}".format(run.info.experiment_id))
        print("status:{}".format(run.info.status))
        print("start_time:{}".format(run.info.start_time))
        print("end_time:{}".format(run.info.end_time))
        print("lifecycle_stage:{}".format(run.info.lifecycle_stage))
    
    
    
    
    
    