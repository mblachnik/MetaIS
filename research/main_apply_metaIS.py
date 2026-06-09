import sklearn.neighbors as knn
#import lightgbm as lgb
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.svm import SVC
#from sklearn.linear_model import LogisticRegression
from research.basics.main_apply_metaIS import applyMetaIS

model = knn.KNeighborsClassifier(n_neighbors=1)
#model = lgb.LGBMClassifier() #LightGBM
#model = RandomForestClassifier(n_estimators=100, random_state=42) #RFC
#model = SVC(kernel='rbf', C=1.0, gamma='scale') #SVM i SVM2
#model = LogisticRegression(max_iter=1000) #LR

applyMetaIS(model)