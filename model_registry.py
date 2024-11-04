import mlflow 
from sklearn.ensemble import RandomForestRegressor

class CustomModel(mlflow.pyfunc.PythonModel):
    
    def predict(self, context, model_input):
        return model_input
    
    
if __name__ == "__main__":
    
    experiment = mlflow.create_experiment(
        'Model_registry',
        artifact_location = "model_registry_artifacts",
        tags = {"env":"dev","version":"1.0.0"},
    )
    
    mlflow.set_experiment(experiment_name="Model_registry")
    
    with mlflow.start_run(run_name="Model_registry") as run:
        model = CustomModel()
        mlflow.pyfunc.log_model(artifact_path="custom_model",python_model=model,registered_model_name="CustomModel")
        mlflow.sklearn.log_model(artifact_path="rfr_model",sk_model=RandomForestRegressor(),registered_model_name="RandomForestRegressor")
        mlflow.sklearn.log_model(artifact_path="rfr_model2",sk_model=RandomForestRegressor())
      