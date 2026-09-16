from dataclasses import dataclass


@dataclass(frozen=True)
class Grid2D:
    nx: int
    ny: int
    length_x: float
    length_y: float

    def __post_init__(self) -> None:
        if self.nx < 3 or self.ny < 3:
            raise ValueError("Grid dimensions must be at least 3x3.")

        if self.length_x <= 0 or self.length_y <= 0:
            raise ValueError("Grid lengths must be positive.")

    @property
    def dx(self) -> float:
        return self.length_x / (self.nx - 1)

    @property
    def dy(self) -> float:
        return self.length_y / (self.ny - 1)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)
