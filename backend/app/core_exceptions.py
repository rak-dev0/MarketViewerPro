from fastapi import HTTPException, status


class UpstreamDataError(HTTPException):
    def __init__(self, detail: str = "Failed to fetch upstream data"):
        super().__init__(status_code=status.HTTP_502_BAD_GATEWAY, detail=detail)
