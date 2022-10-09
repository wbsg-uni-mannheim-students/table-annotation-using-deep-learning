# Download Schema.org Table Corpus tables
# http://webdatacommons.org/structureddata/sotab/

# Download datasets 
wget -P data/zipped-data/ http://data.dws.informatik.uni-mannheim.de/structureddata/sotab/CPA_Training.zip
wget -P data/zipped-data/ http://data.dws.informatik.uni-mannheim.de/structureddata/sotab/CPA_Validation.zip
wget -P data/zipped-data/ http://data.dws.informatik.uni-mannheim.de/structureddata/sotab/CPA_Test.zip
wget -P data/zipped-data/ http://data.dws.informatik.uni-mannheim.de/structureddata/sotab/CTA_Training.zip
wget -P data/zipped-data/ http://data.dws.informatik.uni-mannheim.de/structureddata/sotab/CTA_Validation.zip
wget -P data/zipped-data/ http://data.dws.informatik.uni-mannheim.de/structureddata/sotab/CTA_Test.zip

#Unzip the files into CTA and CPA folders
unzip data/zipped-data/CPA_Training.zip -d data/CPA/
unzip data/zipped-data/CPA_Validation.zip -d data/CPA/
unzip data/zipped-data/CPA_Test.zip -d data/CPA/
unzip data/zipped-data/CTA_Training.zip -d data/CTA/
unzip data/zipped-data/CTA_Validation.zip -d data/CTA/
unzip data/zipped-data/CTA_Test.zip -d data/CTA/