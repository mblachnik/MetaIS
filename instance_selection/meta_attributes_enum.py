from enum import Enum

class MetaAttributesEnum(Enum):

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
    
    def generateColumns(k_values: list[int] = [3,5,9,15,23,33]):
        feature_names = [MetaAttributesEnum.minDistanceSameClass.value, MetaAttributesEnum.minDistanceOppositeClass.value, MetaAttributesEnum.minDistanceAnyClass.value]
        for mat in k_values:
            strMat = str(mat)
            feature_names.append(MetaAttributesEnum.sameClassNeighbors(strMat))
            feature_names.append(MetaAttributesEnum.oppositeClassNeighbors(strMat))
            feature_names.append(MetaAttributesEnum.meanDistanceAnyClass(strMat))
            feature_names.append(MetaAttributesEnum.meanDistanceSameClass(strMat))
            feature_names.append(MetaAttributesEnum.meanDistanceOppositeClass(strMat))

        return feature_names
