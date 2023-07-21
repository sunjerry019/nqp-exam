#!/usr/bin/env python3

# https://stackoverflow.com/a/33533514/3211506
from __future__ import annotations

import enum
import numpy as np
import scipy.sparse
import scipy.linalg
import scipy.sparse.linalg

from typing import Union, Literal, overload


class ST(enum.Enum):
    BRA = 0
    KET = 1


class State:
    def __init__(self, L: int, vector: np.ndarray, typ: ST) -> None:
        self.L = L
        self.vector = vector

        assert typ in ST
        self.type = typ

        if typ == ST.BRA:
            self.vector = self.vector.reshape((1, -1))
        elif typ == ST.KET:
            self.vector = self.vector.reshape((-1, 1))

        self.operator = Operator

    def normalize(self) -> State:
        if not np.any(np.iscomplex(self.vector)):
            self.vector = self.vector.astype(np.longdouble)

        self.vector /= np.linalg.norm(self.vector)
        assert np.isclose(np.linalg.norm(self.vector), 1)

        return self

    def dagger(self) -> State:
        newtype = ST.BRA if self.type == ST.KET else ST.KET
        return type(self)(
            self.L, self.vector.conj(), newtype
        )  # constructor handles the transposing

    # definitions
    def __add__(self, o: Union[Literal, int, float, complex, State]) -> State:
        assert isinstance(o, (int, float, complex, State))

        if isinstance(o, State):
            assert self.L == o.L, "Number of lattices sites do not match"
            newvec = self.vector + o.vector
        else:
            newvec = self.vector + o  # type: ignore

        return type(self)(self.L, newvec, self.type)

    def __sub__(self, o: Union[Literal, int, float, complex, State]) -> State:
        return self.__add__(o * -1)  # type: ignore

    def __mul__(self, o: Union[Literal, int, float, complex]) -> State:
        assert isinstance(o, (int, float, complex))
        newvec = o * self.vector
        return type(self)(self.L, newvec, self.type)

    def __truediv__(self, o: Union[Literal, int, float, complex]) -> State:
        assert isinstance(o, (int, float, complex))
        newvec = 1 / o * self.vector
        return type(self)(self.L, newvec, self.type)

    def __floordiv__(self, o: Union[Literal, int, float, complex]) -> State:
        assert isinstance(o, (int, float, complex))
        newvec = self.vector // o  # type: ignore
        return type(self)(self.L, newvec, self.type)

    def __mod__(self, o: Union[Literal, int, float, complex]) -> State:
        assert isinstance(o, (int, float, complex))
        newvec = self.vector % o  # type: ignore
        return type(self)(self.L, newvec, self.type)

    # in place definitions
    def __iadd__(self, o: Union[Literal, int, float, complex, State]) -> State:
        assert isinstance(o, (int, float, complex, State))

        if isinstance(o, State):
            assert self.L == o.L, "Number of lattices sites do not match"
            self.vector += o.vector
        else:
            self.vector += o  # type: ignore

        return self

    def __isub__(self, o: Union[Literal, int, float, complex, State]) -> State:
        return self.__iadd__(o * -1)  # type: ignore

    def __imul__(self, o: Union[Literal, int, float, complex]) -> State:
        assert isinstance(o, (int, float, complex))
        self.vector = o * self.vector
        return self

    def __itruediv__(self, o: Union[Literal, int, float, complex]) -> State:
        assert isinstance(o, (int, float, complex))
        self.vector = 1 / o * self.vector
        return self

    def __ifloordiv__(self, o: Union[Literal, int, float, complex]) -> State:
        assert isinstance(o, (int, float, complex))
        self.vector = self.vector // o  # type: ignore
        return self

    def __imod__(self, o: Union[Literal, int, float, complex]) -> State:
        assert isinstance(o, (int, float, complex))
        self.vector = self.vector % o  # type: ignore
        return self

    # matmuls
    # https://stackoverflow.com/a/55990630/3211506
    def __imatmul__(self, o: Operator) -> State:
        assert isinstance(o, Operator)

        if self.type == ST.KET:
            raise TypeError("Unable to multiply ket with operator")

        self.vector @= o.matrix

        return self

    @overload
    def __matmul__(self, o: State) -> Union[float, complex]:
        pass

    @overload
    def __matmul__(self, o: State) -> Operator:
        pass

    @overload
    def __matmul__(self, o: Operator) -> State:
        pass

    def __matmul__(
        self, o: Union[Operator, State]
    ) -> Union[Operator, State, Union[float, complex]]:
        if self.type == ST.BRA and isinstance(o, Operator):
            return type(self)(self.L, self.vector @ o.matrix, self.type)
        elif self.type == ST.KET and o.type == ST.BRA:
            return self.operator(self.L, self.vector @ o.vector)
        elif self.type == ST.BRA and o.type == ST.KET:
            return (self.vector @ o.vector).flatten()[0]

        raise TypeError(f"MatMul not defined for {self.type} and {o}")

    # Comparisons
    def __eq__(self, o: State) -> Union[bool, np.bool_]:
        assert isinstance(o, State)
        return (self.L == o.L) and np.all(self.vector == o.vector)

    def expand_to(self, newL: int, before: list[State], after: list[State]) -> State:
        raise NotImplementedError("Not implemented")


class Operator:
    def __init__(self, L: int, matrix: np.ndarray, sparse=True) -> None:
        self.L = L
        if sparse and not scipy.sparse.issparse(matrix):
            matrix = scipy.sparse.coo_matrix(matrix).tocsr()

        self.sparse = scipy.sparse.issparse(matrix)
        self.matrix = matrix

    def dagger(self) -> Operator:
        return type(self)(self.L, self.matrix.getH())

    def __add__(self, o: Union[Literal, int, float, complex, Operator]) -> Operator:
        assert isinstance(o, (int, float, complex, Operator))

        if isinstance(o, Operator):
            assert self.L == o.L, "Number of lattices sites do not match"
            newvec = self.matrix + o.matrix
        else:
            newvec = self.matrix + o  # type: ignore

        return type(self)(self.L, newvec)

    def __sub__(self, o: Union[Literal, int, float, complex, Operator]) -> Operator:
        return self.__add__(o * -1)  # type: ignore

    def __mul__(self, o: Union[Literal, int, float, complex]) -> Operator:
        assert isinstance(o, (int, float, complex))
        newvec = o * self.matrix
        return type(self)(self.L, newvec)

    def __truediv__(self, o: Union[Literal, int, float, complex]) -> Operator:
        assert isinstance(o, (int, float, complex))
        newvec = 1 / o * self.matrix
        return type(self)(self.L, newvec)

    def __floordiv__(self, o: Union[Literal, int, float, complex]) -> Operator:
        assert isinstance(o, (int, float, complex))
        newvec = self.matrix // o  # type: ignore
        return type(self)(self.L, newvec)

    def __mod__(self, o: Union[Literal, int, float, complex]) -> Operator:
        assert isinstance(o, (int, float, complex))
        newvec = self.matrix % o  # type: ignore
        return type(self)(self.L, newvec)

    # in place definitions
    def __iadd__(self, o: Union[Literal, int, float, complex, Operator]) -> Operator:
        assert isinstance(o, (int, float, complex, Operator))

        if isinstance(o, Operator):
            assert self.L == o.L, "Number of lattices sites do not match"
            self.matrix += o.matrix
        else:
            self.matrix += o  # type: ignore

        return self

    def __isub__(self, o: Union[Literal, int, float, complex, Operator]) -> Operator:
        return self.__iadd__(o * -1)  # type: ignore

    def __imul__(self, o: Union[Literal, int, float, complex]) -> Operator:
        assert isinstance(o, (int, float, complex))
        self.matrix = o * self.matrix
        return self

    def __itruediv__(self, o: Union[Literal, int, float, complex]) -> Operator:
        assert isinstance(o, (int, float, complex))
        self.matrix = 1 / o * self.matrix
        return self

    def __ifloordiv__(self, o: Union[Literal, int, float, complex]) -> Operator:
        assert isinstance(o, (int, float, complex))
        self.matrix = self.matrix // o  # type: ignore
        return self

    def __imod__(self, o: Union[Literal, int, float, complex]) -> Operator:
        assert isinstance(o, (int, float, complex))
        self.matrix = self.matrix % o  # type: ignore
        return self

    # matmuls
    # https://stackoverflow.com/a/55990630/3211506
    def __imatmul__(self, o: Operator) -> Operator:
        assert isinstance(o, Operator)

        self.matrix @= o.matrix

        return self

    # https://stackoverflow.com/a/52449229/3211506
    @overload
    def __matmul__(self, o: State) -> State:
        pass

    @overload
    def __matmul__(self, o: Operator) -> Operator:
        pass

    def __matmul__(self, o: Union[Operator, State]) -> Union[Operator, State]:
        assert isinstance(o, (Operator, State))

        if isinstance(o, Operator):
            newvec = self.matrix @ o.matrix
            return type(self)(self.L, newvec)

        # Its a state
        if o.type == ST.BRA:
            raise TypeError("Unable to multiply operator with bra")

        newvec = self.matrix @ o.vector
        return type(o)(o.L, newvec, o.type)

    def expand_to(self, newL: int, site: int) -> Operator:
        raise NotImplementedError("Not implemented")

    @staticmethod
    def ishermitian(sparse: bool, matrix: Union[np.ndarray, scipy.linalg.sparse.csc_matrix]) -> bool:
        if sparse:
            return Operator.ishermitian(sparse = False, matrix = matrix.todense())

        try:
            return scipy.linalg.ishermitian(matrix)
        except AttributeError as e:
            # We are on the cip pool with python 3.9 and scipy 1.3.3
            # https://github.com/scipy/scipy/issues/7308#issue-222144044
            return np.allclose(matrix, np.asmatrix(matrix).getH())


    def get_eigvals(self):
        hermitian = Operator.ishermitian(self.sparse, self.matrix)

        if not self.sparse:
            if hermitian:
                return np.real(np.sort(np.linalg.eigvalsh(self.matrix)))
            else:
                return np.sort(np.linalg.eigvals(self.matrix))
        else:
            if hermitian:
                eigval, eigvec = scipy.sparse.linalg.eigsh(
                    self.matrix, which="LM", k=self.matrix.shape[0] - 1
                )
            else:
                eigval, eigvec = scipy.sparse.linalg.eigs(
                    self.matrix, which="LM", k=self.matrix.shape[0] - 1
                )

            return eigval

    def get_eigsys(self):
        hermitian = Operator.ishermitian(self.sparse, self.matrix)

        if not self.sparse:
            if hermitian:
                return np.linalg.eigh(self.matrix)
            else:
                return np.linalg.eig(self.matrix)
        else:
            if hermitian:
                eigval, eigvec = scipy.sparse.linalg.eigsh(
                    self.matrix, which="LM", k=self.matrix.shape[0] - 1
                )
            else:
                eigval, eigvec = scipy.sparse.linalg.eigs(
                    self.matrix, which="LM", k=self.matrix.shape[0] - 1
                )

            return eigval, eigvec


class HBFockState(State):
    def __init__(self, L: int, vector: np.ndarray, typ: ST) -> None:
        """Hardcore Boson Fock State

        Args:
            L (int): Number of ring sites
            vector (np.ndarray): Vector repr
            typ (ST): ST.KET or ST.BRA
        """

        # https://stackoverflow.com/a/70717629
        if vector.any():
            vector = np.hstack(vector)
        else:
            vector = vector.flatten()

        # https://stackoverflow.com/a/3338368
        if isinstance(vector, np.matrix):
            vector = np.squeeze(np.asarray(vector))

        assert len(vector) == 2 ** (L + 1), "Fock State vector must be 2**(L+1)"
        super().__init__(L, vector, typ)
        self.operator = HBFockOperator

    @staticmethod
    def from_fock_repr(vector: np.ndarray, typ: ST) -> HBFockState:
        zero = np.array([1, 0])
        one = np.array([0, 1])

        statevectors = [zero, one]

        state = statevectors[vector[0]]
        L = len(vector)
        for i in range(1, L):
            state = np.kron(state, statevectors[vector[i]])

        st = HBFockState(L - 1, state, typ)
        st.normalize()

        return st

    def expand_to(
        self, newL: int, before: list[HBFockState], after: list[HBFockState]
    ) -> HBFockState:
        vector = None
        for i, state in enumerate(before):
            if state.type != self.type:
                state = state.dagger()

            if i == 0:
                vector = before[0].vector
            else:
                vector = np.kron(vector, state.vector)  # type: ignore

        vector = np.kron(vector, self.vector)  # type: ignore

        for state in enumerate(after):
            vector = np.kron(vector, state.vector)  # type: ignore

        return HBFockState(newL, vector, self.type)


class HBFockOperator(Operator):
    def __init__(
        self, L: int, matrix: Union[np.ndarray, None] = None, *args, **kwargs
    ) -> None:
        if matrix is None:
            matrix = np.zeros((2 ** (L + 1), 2 ** (L + 1)))

        shape = matrix.shape
        assert shape == (
            2 ** (L + 1),
            2 ** (L + 1),
        ), "Fock Operator must be 2**(L+1) x 2**(L+1)"
        super().__init__(L, matrix, *args, **kwargs)

    def expand_to(self, newL: int, site: int) -> HBFockOperator:
        if not self.sparse:
            op = np.kron(np.eye(2**site, 2**site), self.matrix)
            op = np.kron(op, np.eye(2 ** (newL - site), 2 ** (newL - site)))
        else:
            op = scipy.sparse.kron(scipy.sparse.eye(2**site, 2**site), self.matrix)
            op = scipy.sparse.kron(
                op, scipy.sparse.eye(2 ** (newL - site), 2 ** (newL - site))
            )

        return HBFockOperator(newL, op)
