# TaxEL

## Datasets

- [ncbi-disease](https://github.com/dmis-lab/BioSyn)
- [bc5cdr-disease](https://github.com/dmis-lab/BioSyn)
- [bc5cdr-chemical](https://github.com/dmis-lab/BioSyn)
- [COMETA-clinic](https://drive.google.com/file/d/1bm_b1dwJYxp3vbMw7vc05-CFWD61JyrF/view?usp=drive_link)
- [AAP](https://drive.google.com/file/d/18VQ6LxSbv8Q4TboTHjeX4DFDAXWd3JLD/view?usp=drive_link)


## Taxonomy

- [CTD-Chemical](https://drive.google.com/file/d/1Q8cVl2L-A15sIujKu8e0uu-BZmvHhWqG/view?usp=drive_link)
    - Download from [CTD](https://web.archive.org/web/20180108033447/http://ctdbase.org/downloads)
    - Used for bc5cdr-chemical
- [CTD-Disease](https://drive.google.com/file/d/1BMo38fPwhDWNtb3AHW1GsQFVn8s7dZzD/view?usp=drive_link)
    - Download from [CTD](https://web.archive.org/web/20180108033447/http://ctdbase.org/downloads)
    - Used for bc5cdr-disease and ncbi-disease
- [SNOMED-CT](https://drive.google.com/file/d/1QkqAyZzvknigxQKrAouwLaM0ZiPyFYG-/view?usp=drive_link)
    - Download from [SNOMED-CT](https://www.nlm.nih.gov/healthit/snomedct/index.html?_gl=1*z0twj7*_ga*MzQ4OTkzNTEyLjE2NTYzOTg1Nzc.*_ga_P1FPTH9PL4*MTczMjg1Mzg3Mi40OC4wLjE3MzI4NTM4NzcuMC4wLjA.*_ga_7147EPK006*MTczMjg1Mzg3Mi40Ny4wLjE3MzI4NTM4NzcuMC4wLjA.) 
    - Used for AAP and COMETA


## Train
~~~bash
cd TaxEL
bash scripts/train/ncbi_disease.sh
~~~


## Evaluation

~~~bash
cd TaxEL
bash scripts/eval/ncbi_disease.sh
~~~    


<!-- ### Trained models

You can directly use our trained model for evaluation and prediction.

- [ncbi-disease](https://huggingface.co/TCMLLM/CLOnEL-NCBI-Disease)
- [bc5cdr-disease](https://huggingface.co/TCMLLM/CLOnEL-BC5CDR-Disease)
- [bc5cdr-chemical](https://huggingface.co/TCMLLM/CLOnEL-BC5CDR-Chemical)
- [cometa-cf](https://huggingface.co/TCMLLM/CLOnEL-COMETA-CF)
- [aap](https://huggingface.co/TCMLLM/CLOnEL-AAP) -->