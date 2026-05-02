import shlex
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class CommandInvocation:
    name: str
    args: List[str] = field(default_factory=list)
    flags: Dict[str, str] = field(default_factory=dict)
    raw: str = ""

def parse_command(text: str) -> CommandInvocation:
    text = text.strip()
    if not text.startswith("/"):
        return CommandInvocation(name="", raw=text)
    
    parts = shlex.split(text[1:])
    if not parts:
        return CommandInvocation(name="", raw=text)
        
    name = parts[0]
    args = []
    flags = {}
    
    i = 1
    while i < len(parts):
        part = parts[i]
        if part.startswith("--"):
            key = part[2:]
            if i + 1 < len(parts) and not parts[i+1].startswith("-"):
                flags[key] = parts[i+1]
                i += 2
            else:
                flags[key] = "true"
                i += 1
        elif part.startswith("-"):
            key = part[1:]
            if i + 1 < len(parts) and not parts[i+1].startswith("-"):
                flags[key] = parts[i+1]
                i += 2
            else:
                flags[key] = "true"
                i += 1
        else:
            args.append(part)
            i += 1
            
    return CommandInvocation(name=name, args=args, flags=flags, raw=text)
