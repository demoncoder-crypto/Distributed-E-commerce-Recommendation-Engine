import ray
from ray import train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from ml.models.hybrid_model import HybridRecModel

def train_func(config):
    dataset = train.get_dataset_shard("train")
    model = HybridRecModel(1e6, 5e5, 512)
    model = train.torch.prepare_model(model)
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"])
    
    for epoch in range(10):
        for batch in dataset.iter_torch_batches(batch_size=1024):
            user_ids = batch["user_id"]
            item_ids = batch["item_id"]
            # ... (full training logic)
            train.report({"loss": loss.item()})

trainer = TorchTrainer(
    train_func,
    scaling_config=ScalingConfig(num_workers=4, use_gpu=True)
)
result = trainer.fit()