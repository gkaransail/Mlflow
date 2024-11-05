import mlflow 
from mlflow import MlflowClient

if __name__ == "__main__":
    
    experiment = mlflow.create_experiment(
        'Model_registry',
        artifact_location = "testing_mlflow1_artifacts",
        tags = {"env":"dev","version":"1.0.0"},
    )
    
    mlflow.set_experiment(experiment_name="Model_registry")
    
    
    client = MlflowClient()
    model_name = "registered_model_2"
    
    #create registered model
    client.create_registered_model(model_name)
    
    
    #create model version 
    source = "file:///C:/ETL/Airflow/model_registry_artifacts/22bcbc1ba66a4bdbafaa3d60a2653cc6/artifacts/rfr_model2"
    run_id = "22bcbc1ba66a4bdbafaa3d60a2653cc6"
    client.create_model_version(name=model_name,source=source,run_id=run_id)
    
    
    #transition model version stage
    
    client.transition_model_version_stage(name=model_name, version=2,stage="Staging")