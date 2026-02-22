class DocumentLoader:
    def __init__(self, file_path: str):
        self._file_path = file_path

    def load(self) -> str:
        with open(self._file_path, "r", encoding="utf-8") as file:
            return file.read()
