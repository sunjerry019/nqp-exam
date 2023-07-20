#!/usr/bin/env python3

# https://stackoverflow.com/a/33533514/3211506
from __future__ import annotations

import enum
import numpy as np

from typing import Union, Literal

from QM import State

class ST(enum.Enum):
    BRA = 0
    KET = 1

class State():
    def __init__(self, L: int, vector: np.ndarray, typ: ST) -> None:
        self.L = L
        self.vector = vector 

        assert typ in ST
        self.type = typ

        if typ == ST.BRA:
            self.vector = self.vector.reshape((1, -1))
        elif typ == ST.KET:
            self.vector = self.vector.reshape((-1, 1))

    def dagger(self) -> State:
        newtype = ST.BRA if self.type == ST.KET else ST.KET
        return type(self)(self.L, self.vector, newtype)
    
    # definitions
    def __add__(self, o: Union[Literal, int, float, complex, State]) -> State:
        assert isinstance(o, (Literal, int, float, complex, State))

        if isinstance(o, State):
            assert self.L == o.L, "Number of lattices sites do not match"
            newvec = self.vector + o.vector
        else:
            newvec = self.vector + o # type: ignore

        return type(self)(self.L, newvec, self.type)
    
    def __sub__(self, o: Union[Literal, int, float, complex, State]) -> State:
        return self.__add__(o * -1) # type: ignore
    
    def __mul__(self, o: Union[Literal, int, float, complex]) -> State:
        assert isinstance(o, (int, float, complex))
        newvec = o * self.vector
        return type(self)(self.L, newvec, self.type)
    
    def __truediv__(self, o: Union[Literal, int, float, complex]) -> State:
        assert isinstance(o, (int, float, complex))
        newvec = 1/o * self.vector
        return type(self)(self.L, newvec, self.type)
    
    def __floordiv__(self, o: Union[Literal, int, float, complex]) -> State:
        assert isinstance(o, (int, float, complex))
        newvec = self.vector // o # type: ignore
        return type(self)(self.L, newvec, self.type)
    
    def __mod__(self, o: Union[Literal, int, float, complex]) -> State:
        assert isinstance(o, (int, float, complex))
        newvec = self.vector % o # type: ignore
        return type(self)(self.L, newvec, self.type)
    
    # in place definitions
    def __iadd__(self, o: Union[Literal, int, float, complex, State]) -> State:
        assert isinstance(o, (Literal, int, float, complex, State))

        if isinstance(o, State):
            assert self.L == o.L, "Number of lattices sites do not match"
            self.vector += o.vector
        else:
            self.vector += o # type: ignore

        return self
    
    def __isub__(self, o: Union[Literal, int, float, complex, State]) -> State:
        return self.__iadd__(o * -1) # type: ignore
    
    def __imul__(self, o: Union[Literal, int, float, complex]) -> State:
        assert isinstance(o, (int, float, complex))
        self.vector = o * self.vector
        return self
    
    def __itruediv__(self, o: Union[Literal, int, float, complex]) -> State:
        assert isinstance(o, (int, float, complex))
        self.vector = 1/o * self.vector
        return self
    
    def __ifloordiv__(self, o: Union[Literal, int, float, complex]) -> State:
        assert isinstance(o, (int, float, complex))
        self.vector = self.vector // o # type: ignore
        return self
    
    def __imod__(self, o: Union[Literal, int, float, complex]) -> State:
        assert isinstance(o, (int, float, complex))
        self.vector = self.vector % o # type: ignore
        return self

    # matmuls
    # https://stackoverflow.com/a/55990630/3211506
    def __imatmul__(self, o: Operator) -> State:
        assert isinstance(o, Operator)

        if self.type == ST.KET:
            raise TypeError("Unable to multiply ket with operator")
        
        self.vector @= o.matrix

        return self
    
    def __matmul__(self, o: Operator) -> State:
        assert isinstance(o, Operator)

        if self.type == ST.KET:
            raise TypeError("Unable to multiply ket with operator")
        
        newvec = self.vector @ o.matrix

        return type(self)(self.L, newvec, self.type)
    
    # Comparisons    
    def __eq__(self, o: State) -> Union[bool, np.bool_]:
        assert isinstance(o, State)
        return (self.L == o.L) and np.all(self.vector == o.vector)
    
    def expand_to(self, newL: int) -> State:
        raise NotImplementedError("Not implemented")
    
class Operator():
    def __init__(self, L: int, matrix: np.ndarray) -> None:
        self.L = L
        self.matrix = matrix

class FockState(State):
    def __init__(self, L: int, vector: np.ndarray, typ: ST) -> None:
        assert len(vector) == L, "Fock State vector must be the same length as the number of sites"
        super().__init__(L, vector, typ)

    def expand_to(self, newL: int) -> FockState:
        return self

    # def to_product_state(self) -> ProductState:

# class ProductState(State):
#     def __init__(self, L: int, vector: np.ndarray = None) -> None:
#         self.


