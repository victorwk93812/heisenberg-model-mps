class DimDismatchError(Exception):
    def __init__(self, dim1, dim2):
        message = f"Values are unequal: {dim1} and {dim2}"
        super().__init__(message)
        self.dim1 = dim1
        self.dim2 = dim2


