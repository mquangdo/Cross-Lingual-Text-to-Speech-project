import wandb
 
run = wandb.init(
    project="sparktts",
    job_type="upload_checkpoint",
    name="upload-final-checkpoint"
)

artifact = wandb.Artifact(
    name="spartk-tts-llm-0.5B",
    type="model"
)

artifact.add_dir("outputs_vietmuong/out")

run.log_artifact(artifact)

run.finish()