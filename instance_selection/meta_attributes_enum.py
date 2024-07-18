from enum import Enum

class MetaAttributesEnum(Enum):

    id = 'id'

    minDistanceSameClass = 'minDistanceSameClass'

    minDistanceOppositeClass = 'minDistanceOppositeClass'

    minDistanceAnyClass = 'minDistanceAnyClass'

    meanDistanceAny = 'meanDistanceAny'

    def sameClassNeighbors(k):
        return 'sameClassNeighbors' + k
    
    def oppositeClassNeighbors(k):
        return 'oppositeClassNeighbors' + k
    
    def meanDistanceAnyClass(k):
        return 'meanDistanceAnyClass' + k
    
    def meanDistanceSameClass(k):
        return 'meanDistanceSameClass' + k
    
    def meanDistanceOppositeClass(k):
        return 'meanDistanceOppositeClass' + k
