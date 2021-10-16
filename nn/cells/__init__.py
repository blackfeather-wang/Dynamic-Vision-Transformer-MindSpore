"""Some additional Cell inheritors."""

# pylint: disable = arguments-differ

from mindspore.nn import Cell

class ResidualCell(Cell):
    """Cell which implements x + f(x) function."""
    def __init__(self, cell):
        super().__init__()
        self.cell = cell

    def construct(self, x, **kwargs):
        return self.cell(x, **kwargs) + x
