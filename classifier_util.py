import xgboost as xgb
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import argparse
import sys
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--drugs", "-d", nargs='*', required=True, help="List of drugs")
parser.add_argument("--phases", "-p", nargs='*', required=True, help="List of phases")
args = parser.parse_args()

# classifier = xgb.XGBClassifier()
# booster = xgb.Booster()
# booster.load_model('001.model')
# classifier._Booster = booster
# classifier._le = LabelEncoder().fit([0, 1])

with open('LR_model.pkl', 'rb') as fid:
    LR = pickle.load(fid)

data = pd.read_pickle('util_data.pkl')
data['Interventions2'] = [args.drugs] * len(data)
data['Phases2'] = [args.phases] * len(data)

drugs_binarizer = MultiLabelBinarizer()
phase_binarizer = MultiLabelBinarizer()

try:
    data_new = pd.DataFrame(phase_binarizer.fit_transform(data.Phases),
                            columns='Is 1st phase ' + phase_binarizer.classes_,
                            index=data.index)
    data_new = data_new.join(
        pd.DataFrame(drugs_binarizer.fit_transform(data.Interventions), columns='Is 1st ' + drugs_binarizer.classes_,
                     index=data.index))
    data_new = data_new.join(
        pd.DataFrame(phase_binarizer.transform(data.Phases2), columns='Is 2nd phase  ' + phase_binarizer.classes_,
                     index=data.index))
    data_new = data_new.join(
        pd.DataFrame(drugs_binarizer.transform(data.Interventions2), columns='Is 2nd ' + drugs_binarizer.classes_,
                     index=data.index))
except KeyError:
    print('Incorrect arguments')
    sys.exit()

data['Is_similar'] = classifier.predict(np.array(data_new))

result = []

for i in data.index:
    if data['Is_similar'][i] == 1:
        result.append(data['NCT Number'][i])

print(result)

