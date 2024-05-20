from enum import Enum

class RunMode(Enum):
    Other = 0
    FullSets = 1
    FoldsOnly = 2
    TestRun = 3
    Random = 4

class ProcessStep(Enum):
    Other = 0
    GenerateMetadata = 1
    ConnectMetaWithProto = 2
    TestRun = 3
    Random = 4
