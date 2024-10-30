import mlflow 
from ml_flow import get_mlflow_experiment


if__name__=="__main__":
    
    experiment = get_mlflow_experiment(experiment_id="432907354826472324")
    
    print("Name:{}".format(experiment.name))
    print("Experiment_id:{}".format(experiment.experiment_id))
    print("Artifact Location:{}".format(experiment.artifact_location))
    print("Tags:{}".format(experiment.tags))
    print("Lifecycle_stage:{}".format(experiment.lifecycle_stage))
    print("Creation timestamp:{}".format(experiment.creation_time))