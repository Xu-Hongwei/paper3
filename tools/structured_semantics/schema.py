from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any


@dataclass
class Attribute:
    type: str
    value: str


@dataclass
class Entity:
    id: str
    text: str
    attributes: List[Attribute] = field(default_factory=list)


@dataclass
class Relation:
    subject: str
    predicate: str
    object: str


@dataclass
class StructuredSemantics:
    entities: List[Entity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)