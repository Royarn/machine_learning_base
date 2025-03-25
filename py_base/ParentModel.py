class ParentModel:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        print("Calling ParentModel __call__")
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        print("Calling ParentModel forward")

    def alone(self):
        print("Calling ParentModel alone")
