from pydantic import BaseModel, Field
from typing import Annotated, List
from enum import IntFlag


class mc_output_model(BaseModel):
    question: str
    alternatives: Annotated[list[str], Field(min_length=4, max_length=5)]
    correct_answer: Annotated[str, Field(max_length=1)]
    justify: str
                        
class tf_output_model(BaseModel):
    question: str
    alternatives: Annotated[list[str], Field(min_length=2, max_length=6)]
    answers: Annotated[list[bool], Field(min_length=2, max_length=6)]
    justify: Annotated[list[str], Field(min_length=2, max_length=6)]

class d_output_model(BaseModel):
    question: str
    answer: str

output_model = {
        1: (mc_output_model, "MULTIPLE_CHOICE"),
        2: (tf_output_model, "TRUE_OR_FALSE"),
        3: (d_output_model, "DISCURSIVE"),
}
