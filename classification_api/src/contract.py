from pydantic import BaseModel


class Response(BaseModel):
    disease: int
