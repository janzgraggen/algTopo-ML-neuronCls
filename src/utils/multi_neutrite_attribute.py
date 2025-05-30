class MultiNeutriteAttribute:
    """
    usage if x = MultiNeutriteAttribute()
    x.axon = torch.tensor(x) with x being the x of the axon
    x.apical = torch.tensor(x) with x being the x of the apical
    x.basal = torch.tensor(x) with x being the x of the basals


    """
    def __init__(self, attr_axon=None, attr_apical=None, attr_basal=None):
        self.axon = attr_axon
        self.apical = attr_apical
        self.basal = attr_basal

    def _preferable_neurite(self):
        return next(attr for attr in [self.axon, self.apical, self.basal] if attr is not None)

    def size(self, dim):
        print(type(self.axon))
        tensor = self._preferable_neurite()
        return tensor.size(dim)
        

    def __repr__(self):
        return f"MultiNeutriteAttribute(axon={self.axon}, apical={self.apical}, basal={self.basal})"
