import wandb

run = wandb.init(
    project="hifigan",
    job_type="upload_checkpoint",
    name="upload-final-checkpoint"
)

artifact = wandb.Artifact(
    name="hifigan_200epochs",
    type="model"
)

artifact.add_dir("work_dir/hifigan")

run.log_artifact(artifact)

run.finish()