import torch

class ModelInspector:

    def __init__(self, model):
        self.model = model

    def summary(
        self,
        input_szie: tuple[int] | None = None,
        input_data: torch.Tensor | None = None,
        batch_dim: int | None = None,
    ):
        import torchinfo
        return torchinfo.summary(
            self.model,
            input_size=input_szie,
            input_data=input_data,
            batch_dim=batch_dim,
        )

    def visualize(self, data, filename: str | None = None):
        if filename is None:
            if hasattr(self.model, "name"):
                filename = self.model.name
            else:
                filename = self.model.__class__.__name__
        import torchview
        return torchview.draw_graph(self.model, input_data=data, filename=filename).visual_graph
