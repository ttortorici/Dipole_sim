class Parent:
    def __init__(self):
        print(self.method1())

    def method1(self):
        print("inside Parent")
        return "Parent"


class Child(Parent):
    def __int__(self):
        super(Parent).__init__()

    def method1(self):
        Parent.method1(self)
        return "Child"


if __name__ == "__main__":
    Child()