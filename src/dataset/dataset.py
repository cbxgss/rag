class Item:
    def __init__(self, id:str, question:str, golden_answers:list[str], metadata:dict):
        self.id: str = id
        self.question: str = question
        self.golden_answers: list[str] = golden_answers
        self.metadata: dict = metadata

    def __getattr__(self, attr_name):
        if attr_name in ['id','question','golden_answers','metadata']:
            return super().__getattribute__(attr_name)
        else:
            raise AttributeError(f"Attribute `{attr_name}` not found")


class Dataset:
    def __init__(self, name, data: list[Item]):
        self.name = name
        self.data: list[Item] = data

    def __getattr__(self, attr_name):
        return [item.__getattr__(attr_name) for item in self.data]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
