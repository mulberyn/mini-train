
class Logger:
    def __init__(
        self,
        use_wandb: bool = False,
        project: str | None = None,
        run_name: str | None = None,
        config: dict | None = None,
    ):
        self.use_wandb = use_wandb
        self.wandb = None
        if use_wandb:
            try:
                import wandb
            except ImportError as e:
                raise ImportError("wandb is required when use_wandb=True") from e
            self.wandb = wandb
            wandb.init(project=project, name=run_name, config=config)


    def log(self, metrics: dict, step: int):
        if self.use_wandb:
            self.wandb.log(metrics, step=step)


    def finish(self):
        if self.use_wandb:
            self.wandb.finish()