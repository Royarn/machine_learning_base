from typing import Tuple


class Demo:
    def __init__(self, param1: str, param2: str):
        print("Initialize Demo instance")
        self.param1 = param1
        self.param2 = param2

    def __call__(self) -> None:
        print("Call Demo instance")

    def str(self) -> str:
        print("__str__ Demo instance")
        return f"Demo(param1={self.param1}, param2={self.param2})"

    # 写一个语法糖函数
    def __myd__(self) -> str:
        print("__myd__ Demo instance")
        return f"Demo(param1={self.param1}, param2={self.param2})"


def test_fn(model_name: str, loss_type: str) -> Tuple["Demo", str, str]:
    return Demo("狗屁", "Nuk"),model_name, loss_type

# 调用函数并解包元组
# test_fn("hello", "MSE")
# test_fn("hello", "MSE")
# demo_obj, model_name, loss_type = test_fn("hello", "MSE")

demo = Demo("para1", "para2")
demo()
demo.str()
myd()