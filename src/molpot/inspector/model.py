import torchviz, torchinfo

class ModelInspector:

    def __init__(self, model):
        self.model = model

    def summary(self):
        torchinfo.summary(self.model)

    def visualize(self, data):
        torchviz.make_dot(self.model(data)).render(self.model.name, format="png", cleanup=True)
