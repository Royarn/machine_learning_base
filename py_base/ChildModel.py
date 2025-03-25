from ParentModel import ParentModel

class ChildModel(ParentModel):
    def __init__(self):
        super(ChildModel, self).__init__()
        # 将父类的forward方法赋值给childMethod
        self.childMethod = ParentModel.alone(self) # 将 child_forward 方法赋值给 childMethod

    def child_forward(self, x):
        return f"Processed by child_forward with input: {x}"

    def forward(self, x):
        return self.childMethod(x)


# 创建子类实例
model = ChildModel()

# 调用子类实例，实际上是在调用父类的 __call__ 方法
input_data = 3
output = model(input_data)  # 这相当于调用 model.__call__(input_data)
print(output)