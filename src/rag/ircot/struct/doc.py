class Doc:
    def __init__(self, id: int, title_content: str):
        self.id = id
        title, content = title_content.split("\n", 1)
        self.title = title.strip().strip("#### ").strip("\"")
        self.content = content.strip()

    def __str__(self):
        return f"#### {self.title}\n\n{self.content}"


class Docs:
    def __init__(self, docs: list[Doc] = None) -> None:
        self.docs: dict[int, Doc] = {} if docs is None else {doc.id: doc for doc in docs}

    def __str__(self):
        return "\n\n".join([str(doc) for doc in self.docs.values()])

    def __len__(self) -> int:
        return len(self.docs)

    def __getitem__(self, i: int) -> Doc:
        id = self.ids()[i]
        if isinstance(i, slice):
            return Docs({id: self.docs[id] for id in self.ids()[i]})
        return self.docs[id]

    def ids(self) -> list[int]:
        return list(self.docs.keys())

    def add(self, docs: list[Doc]) -> "Docs":
        docs_new = {}
        for doc in docs:
            if doc.id not in self.docs:
                docs_new[doc.id] = doc
        self.docs.update(docs_new)
        return Docs(docs_new)
