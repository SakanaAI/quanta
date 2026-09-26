class QProgramError(RuntimeError):
    """Base error for annotated Q-program failures."""


class DefinitionError(QProgramError):
    """A quantum or primitive definition violates the language contract."""


class TraceError(QProgramError):
    """A predictive event cannot be traced unambiguously."""


class CompilationError(QProgramError):
    """Traces cannot be compiled into a valid quanta poset."""


class ComplexityError(CompilationError):
    """A quantum exceeds the configured reference-complexity budget."""
