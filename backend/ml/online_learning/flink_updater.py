from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.functions import ProcessFunction
import torch

class ModelUpdater(ProcessFunction):
    def open(self, context):
        self.model = HybridRecModel(...)
        self.optimizer = torch.optim.Adam(self.model.parameters())
    
    def process_element(self, event, ctx):
        # Update model with event data
        self.optimizer.zero_grad()
        loss = self.model(event)
        loss.backward()
        self.optimizer.step()

env = StreamExecutionEnvironment.get_execution_environment()
env.add_source(KafkaSource(...)) \
   .process(ModelUpdater()) \
   .add_sink(KafkaSink(...))
env.execute()