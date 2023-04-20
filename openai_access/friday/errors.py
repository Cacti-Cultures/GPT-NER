# encoding: utf-8
"""
@author: Xiaofei Sun
@contact: adoni1203@gmail.com
@time: 2023/01/09
@desc: 这只飞很懒
"""
from enum import Enum


class ErrorType(Enum):
    """
    抄过来的，改改改
    """
    timeout = "Timeout"
    no_key = "No Key"
    fail_after_retrying = "Fail After Retrying"
    unknown = "Unknown Error"


class ApiConnectionError(Exception):
    def __init__(self, *, request_id, message, error_type):
        super(ApiConnectionError, self).__init__(message)
        self.request_id = request_id
        self.message = message
        self.error_type = error_type

    def __str__(self):
        return f"Request {self.request_id}: {self.error_type}-{self.message}"

    def __repr__(self):
        return f"ApiConnectionError({self.request_id}, {self.message}, {self.error_type})"


class ApiConnectionTimeoutError(ApiConnectionError):
    def __init__(self, request_id, message):
        super().__init__(request_id=request_id, message=message, error_type=ErrorType.timeout)


class ApiConnectionNoKeyError(ApiConnectionError):
    def __init__(self, request_id, message):
        super().__init__(request_id=request_id, message=message, error_type=ErrorType.no_key)


class ApiConnectionFailAfterRetryingError(ApiConnectionError):
    def __init__(self, request_id, message):
        super().__init__(request_id=request_id, message=message, error_type=ErrorType.fail_after_retrying)


class ApiConnectionUnknownError(ApiConnectionError):
    def __init__(self, request_id, message):
        super().__init__(request_id=request_id, message=message, error_type=ErrorType.unknown)
