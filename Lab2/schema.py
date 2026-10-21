"""Strict structured output with evidence and stable entity identifiers."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

EntityType = Literal['vehicle', 'part', 'symptom', 'procedure', 'criterion']
RelationType = Literal['HAS_PART', 'HAS_SYMPTOM', 'CHECKED_BY', 'REQUIRES', 'SELECTED_BY', 'PRECEDES']

ALLOWED_ENDPOINTS = {
    'HAS_PART': {('vehicle', 'part')},
    'HAS_SYMPTOM': {('part', 'symptom')},
    'CHECKED_BY': {('symptom', 'procedure')},
    'SELECTED_BY': {('part', 'criterion')},
    'REQUIRES': {('procedure', 'criterion'), ('procedure', 'part'), ('procedure', 'procedure')},
    'PRECEDES': {('procedure', 'procedure'), ('criterion', 'procedure')},
}

class Entity(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str = Field(description='Stable lowercase identifier, e.g. part_brake_pad')
    name: str
    type: EntityType

class Fact(BaseModel):
    model_config = ConfigDict(extra='forbid')
    subject: Entity
    relation: RelationType
    object: Entity
    evidence: str = Field(description='Exact verbatim contiguous quote from the input document')

    @model_validator(mode='after')
    def endpoints_match_relation(self):
        pair = (self.subject.type, self.object.type)
        if pair not in ALLOWED_ENDPOINTS[self.relation]:
            raise ValueError(f'{self.relation}: invalid endpoint types {pair}; allowed {ALLOWED_ENDPOINTS[self.relation]}')
        return self

class Extraction(BaseModel):
    model_config = ConfigDict(extra='forbid')
    facts: list[Fact] = Field(min_length=0, max_length=10)
