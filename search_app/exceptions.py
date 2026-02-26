from __future__ import annotations


class AppError(Exception):
    """Base application exception with status metadata."""

    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class ValidationError(AppError):
    def __init__(self, message: str):
        super().__init__(message, status_code=400)


class UnauthorizedError(AppError):
    def __init__(self, message: str = "Unauthorized"):
        super().__init__(message, status_code=401)


class RateLimitError(AppError):
    def __init__(self, message: str = "Too many requests"):
        super().__init__(message, status_code=429)
