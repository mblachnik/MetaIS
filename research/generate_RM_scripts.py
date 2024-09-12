from research.basics.utils import loadConfig

config = loadConfig()

models = [
    #'CCIS',
    #'HMNE',
    #'ENN',
    #'Drop3'
    #'Drop3Keel'
    'ICFKeel'
    ]
folds = [0,1,2,3,4,5]
datasets = ['ring']
datasets = config['datasets']
res = []
for dataset in datasets:
    for model in models:
        for fold in folds:
            script = f'start "{model} {fold} " C:\\PROGRA~1\\RapidMiner\\RAPIDM~1\\jre\\bin\\java -Xmx20g -Xms20g -cp C:\\PROGRA~1\\RapidMiner\\RAPIDM~1\\lib\\* com.rapidminer.launcher.CommandLineLauncher "//Projects/2023/MetaIS/do IS by {model}_man" -Mparent_folder={dataset} -Mrepository=//Datasets/KeelTmp/'
            if fold==0:
                script += f" -Mentry_name={dataset}.dat"
            else:
                script += f" -Mentry_name={dataset}-5-{fold}tra.dat"
            res.append(script)

for r in res:
    print(r)