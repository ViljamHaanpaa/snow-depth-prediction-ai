
Tämä projekti hyödyntää historiallista säädataa ja koneoppimismalleja lumensyvyyden ennustamiseen. Data sisältää lämpötila- ja lumensyvyystietoja useilta vuosikymmeniltä, ja mallia voidaan käyttää arvioimaan tulevaa lumensyvyyttä. 

Data on hankittu avoimista lähteistä ja kattaa Tukholman lämpötila- ja lumensyvyystiedot viimeisen 150 vuoden ajalta.



------ Rakenne ------

dataset.py – Sisältää historiallista lumensyvyysdataa.

dummy_dataset.csv – Esimerkkidata testikäyttöön.

printtocsv.py – Muuntaa lämpötiladatan CSV-muotoon.

model.h5 & snow_depth_model.h5 – Esikoulutetut koneoppimismallit lumensyvyyden ennustamiseen.

stockholm_snow_depth.csv & temperature_data.csv – Todelliset säädatan lähteet.

test_tensorflow.py – Testaa TensorFlow-asennuksen.



Käyttö

Varmista, että ympäristössäsi on asennettuna Python ja tarvittavat kirjastot, kuten TensorFlow ja Pandas.

Suorita printtocsv.py muuntaaksesi lämpötiladatan.

Käytä dataset.py-tiedostoa, jos haluat tutkia lumensyvyysdataa.

Lataa ja käytä model.h5 ennustamiseen.

Riippuvuudet

Python 3+

TensorFlow

Pandas

NumPy

