from urllib.request import urlretrieve

all_years = range(1874, 2022)

for year in all_years:
    url = 'http://solarcyclescience.com/AR_Database/g{}.txt'.format(year)
    destination ='sunspot_data/sunspot_{}.txt'.format(year)

    print('Downloading file for the year: {}'.format(year))

    urlretrieve(url, destination)
