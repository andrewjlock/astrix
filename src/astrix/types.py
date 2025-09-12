class ExampleType1:
    """An example type with a name and a value."""

    def __init__(self, name: str, value: int):
        self.name = name
        self.value = value

    def display(self) -> str:
        return f"Exampletype(name={self.name}, value={self.value})"


class ExampleType2:
    """An example type with a name and a value."""

    def __init__(self, name: str, value: int):
        self.name = name    
        self.value = value

    def display(self) -> str:
        return f"Exampletype(name={self.name}, value={self.value})"
