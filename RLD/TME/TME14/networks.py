from utils import Conv1DLayer, AffineCouplingLayer, \
                  AffineLayer, FlowModel


class Glow(FlowModel):
    def __init__(self, prior, dim, n_blocks):
        flows = []
        for _ in range(n_blocks):
            flows.append(AffineLayer(dim))
            flows.append(Conv1DLayer(dim))
            flows.append(AffineCouplingLayer(dim))
        super().__init__(prior, *flows)

