import mlflow

if __name__ == "__main__":
    
    
    experiment = mlflow.create_experiment(
        'testing_mlflow3_run',
        #artifact_location = "testing_mlflow1_artifacts",
        tags = {"env":"dev","version":"1.0.0"},
    )
    
    mlflow.set_experiment(experiment_name="testing_mlflow3_run")
    with mlflow.start_run(run_name="testing") as run:
        
        
        
        parameters = {
            "learning_rate":0.01,
            "epochas":10,
            "batch_size":100,
            "loss_function":"nse",
            "optimizer":"adam"
            
        }
        #Your mchine learning code goes here 
        mlflow.log_params(parameters)
        
        metrics = {
            "mse":0.01,
            "mae":10,
            "rmse":100,
            "r2":0.01
            
        }
        #Your mchine learning code goes here 
        mlflow.log_params(parameters)
        
        mlflow.log_metrics(metrics)
        
        
        
        ###create a text file that says hello world 
        
        with open("hello_world.txt","w") as f:
            f.write("Hello World!")
            
            
        #log the text file as an artifact
        mlflow.log_artifact(local_path="hello_world.txt",artifact_path="hello_world.txt")
        
        #print run info 
        print("run-id:{}".format(run.info.run_id))
        print("experiment_id:{}".format(run.info.experiment_id))
        print("status:{}".format(run.info.status))
        print("start_time:{}".format(run.info.start_time))
        print("end_time:{}".format(run.info.end_time))
        print("lifecycle_stage:{}".format(run.info.lifecycle_stage))
        