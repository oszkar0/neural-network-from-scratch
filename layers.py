class BaseLayer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagate(self, input):
        raise NotImplementedError

    def backward_propagate(self, output_error, learning_rate):
        raise NotImplementedError