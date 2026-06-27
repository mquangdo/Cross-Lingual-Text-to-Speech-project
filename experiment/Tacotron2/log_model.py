import wandb

run = wandb.init(
    project="tacotron2",
    job_type="upload_checkpoint",
    name="upload-final-checkpoint"
)

artifact = wandb.Artifact(
    name="tacotron2_all_checkpoints",
    type="model"
)

artifact.add_dir("tacotron2")

run.log_artifact(artifact)

run.finish()