"""Processing Element (MAC unit) for weight-stationary systolic array."""

from __future__ import annotations

from systolic_array.types import PEPhase


class ProcessingElement:
    """Stateless MAC: stationary weight, pass-through activation and psum."""

    __slots__ = (
        "row",
        "col",
        "weight",
        "phase",
        "act_in",
        "act_out",
        "weight_in",
        "weight_out",
        "psum_in",
        "psum_out",
        "w_coord",
        "a_coord",
        "p_coord",
        "writeback",
    )

    def __init__(self, row: int, col: int) -> None:
        self.row = row
        self.col = col
        self.weight: float | None = None
        self.phase: PEPhase = PEPhase.IDLE
        self.act_in: float | None = None
        self.act_out: float | None = None
        self.weight_in: float | None = None
        self.weight_out: float | None = None
        self.psum_in: float = 0.0
        self.psum_out: float = 0.0
        self.w_coord: list[int] | None = None
        self.a_coord: list[int] | None = None
        self.p_coord: list[int] | None = None
        self.writeback: bool = False

    def reset(self) -> None:
        self.weight = None
        self.phase = PEPhase.IDLE
        self.act_in = None
        self.act_out = None
        self.weight_in = None
        self.weight_out = None
        self.psum_in = 0.0
        self.psum_out = 0.0
        self.w_coord = None
        self.a_coord = None
        self.p_coord = None
        self.writeback = False

    def load_weight(self) -> None:
        self.weight = 1.0
        self.phase = PEPhase.LOAD_WEIGHT

    def load_psum(self, m: int, c: int) -> None:
        self.psum_out = 0.0
        self.p_coord = [m, c]
        self.phase = PEPhase.LOAD_PSUM

    def begin_compute(self) -> None:
        self.phase = PEPhase.COMPUTE

    def step(self, act_in: float | None, psum_in: float) -> tuple[float | None, float]:
        """WS: stationary weight, pass activation right and psum down."""
        self.act_in = act_in
        self.psum_in = psum_in

        if act_in is not None and self.weight is not None:
            self.psum_out = psum_in + act_in * self.weight
            self.act_out = act_in
        else:
            self.psum_out = psum_in
            self.act_out = None

        return self.act_out, self.psum_out

    def step_os(
        self, act_in: float | None, weight_in: float | None
    ) -> tuple[float | None, float | None]:
        """OS: stationary psum, pass activation down and weight right."""
        self.act_in = act_in
        self.weight_in = weight_in

        if act_in is not None and weight_in is not None:
            self.psum_out = self.psum_out + act_in * weight_in

        self.act_out = act_in
        self.weight_out = weight_in
        return self.act_out, self.weight_out
