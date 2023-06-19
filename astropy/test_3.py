# Important Links
# https://exoplanetarchive.ipac.caltech.edu/docs/API_tce_columns.html
# https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=q1_q17_dr24_tce
# https://github.com/UCLAMLRG/planets_ml
# Astro-Net --> https://git.dst.etit.tu-chemnitz.de/external/tf-models/-/tree/resnet-perf-2/research/astronet
# Exoplanet Detection --> https://freesoft.dev/program/149317974,           https://github.com/dinismf/exoplanet_classification_thesis

import pandas as pd

# df = pd.read_csv('q1_q17_dr24_tce_2022.05.11_23.15.21-filtered copy.csv', index_col=False)

df = pd.read_csv('cumulative.csv', index_col=False)
training_labels = df['koi_disposition']
kep_id = df['kepid']

print(len(training_labels))
print(type(training_labels))

# for id in kep_id:
#     if id == '11517719':
#         print(df.iloc)

count = 0
planets = []
for labels in training_labels:
    if labels == 'CONFIRMED':
        planets.append(df['kepid'][count])
    
    count += 1

print(len(planets))
# print(planets)

possibles = [3749365,9651668, 9730163, 11449844, 11517719, 12019440]

for p in possibles:
    if p in planets:
        print('KEP_ID: {} is a planet'.format(p))
    else:
        print("KEP_ID: {} is not a planet".format(p))