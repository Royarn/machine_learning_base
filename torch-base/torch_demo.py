class ParentModel:
    def __init__(self):
        self.parentMethod = self.alone()
        print("ParentModel __init__ method invoked.")

    def alone(self):
        return "This is a method defined in the parent class."

    def forward(self, x):
        print("This is the forward method in the parent class.")
        pass

    def __call__(self, x):
        print("ParentModel __call__ method invoked.")
        self.forward(x)


class ChildModel(ParentModel):
    def __init__(self):
        super().__init__()
        self.b = None
        self.childMethod = self.alone()
        print("ChildModel __init__ method invoked.")

    def child_forward(self, x):
        print("This is the forward method in the child class.")
        pass

    def forward(self, x):
        print("This is the forward method in the child class.")
        pass
    def alone(self):
        return "This is a method defined in the child class."


model = ChildModel()
print(model.childMethod)