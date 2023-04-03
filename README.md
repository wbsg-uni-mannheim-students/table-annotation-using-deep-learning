# table-annotation-using-deep-learning
Team Project HWS2022 Table Annotation using deep learning

## Reproduction

To reproduce the experiments:

* **Build the conda environment**

  The conda environment *tp-dws.yml* is available at the root directory. Install using ```conda env create -f tp-dws.yml```
  
* **Download the data for SOTAB**

  Remain at root directory and execute *download.sh*
  
* **Preprocess data**

  Redirect to respective folders for Column Type Annotation (CTA) and Column Property Annotation (CPA) under *experiments_final_phase/*. Run the create_new_dataset python script to preprocess the respective data ```python create_new_dataset.py```
  
* **Run experiments**

  Example reproduction code is available at *run.py* 

To reproduce the TURL experiments:

* **Download the Wikitables data** 

  Download rom *https://github.com/sunlab-osu/TURL*. Redirect to the respective directory of *experiments_turl/cta* or *experiments_turl/cpa* and execute   *turl_create_cta_pickle.ipynb* or *turl_create_cpa_pickle.ipynb*
 
* **Run experiments**

  The workflow is similar to our workflow for SOTAB benchmark
  
For additional experiments:

* **Subtables model**
  The code is available in *experiments_final_phase/cpa/create_subtables.py* to create the subtables fpr the CPA task
