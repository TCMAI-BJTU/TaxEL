# TaxEL

Website: [http://el.tcmkg.com/](http://el.tcmkg.com/)

## Datasets

- [ncbi-disease](https://github.com/dmis-lab/BioSyn)
- [bc5cdr-disease](https://github.com/dmis-lab/BioSyn)
- [bc5cdr-chemical](https://github.com/dmis-lab/BioSyn)
- [COMETA](https://drive.google.com/file/d/1XMmLWl54aU3SWQgJ_ULOX_LxZ1EBfw-i/view?usp=drive_link)
- [AAP](https://drive.google.com/file/d/13gJRmCACLcUiFeVbXbPsjRGEOnEBeD1u/view?usp=drive_link)


## Taxonomy

- [CTD-Chemical](https://drive.google.com/file/d/1Q8cVl2L-A15sIujKu8e0uu-BZmvHhWqG/view?usp=drive_link)
    - Download from [CTD](https://web.archive.org/web/20180108033447/http://ctdbase.org/downloads)
    - Used for bc5cdr-chemical
- [CTD-Disease](https://drive.google.com/file/d/1BMo38fPwhDWNtb3AHW1GsQFVn8s7dZzD/view?usp=drive_link)
    - Download from [CTD](https://web.archive.org/web/20180108033447/http://ctdbase.org/downloads)
    - Used for bc5cdr-disease and ncbi-disease
- SNOMED-CT
    - Download from [SNOMED-CT](https://www.healthterminologies.gov.au/access-clinical-terminology/access-snomed-ct-au/) 
    - Used for AAP and COMETA


## Train
```bash
cd TaxEL
bash scripts/train/ncbi_disease.sh
```


## Evaluation

```bash
cd TaxEL
CHECKPOINT_PATH="/path/to/checkpoint" bash scripts/eval/ncbi_disease.sh
```    

## Citation

```bibtex
@article{hua2026taxel,
  title={TaxEL: Taxonomy-Enhanced Entity Representation Learning for Biomedical Entity Linking},
  author={Hua, Rui and Liu, Zeyu and Shu, Zixin and Chang, Kai and Ye, Mingwei and Xia, Jianan and Zhou, Xuezhong},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2026},
  publisher={IEEE}
}
```
