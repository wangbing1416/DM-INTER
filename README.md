# DM-INTER
This repo is the released code of our work **WhyMisinformation is Created? Detecting them by Integrating Intent Features**, published in CIKM'24.

🎉 [ 2024/07 ] This paper is accepted by CIKM 2024.

Our released code follows to "Generalizing to the Future: Mitigating Entity Bias in Fake News Detection".

### Requirements

```
torch==1.12.1
cudatoolkit==11.3.1
transformers==4.27.4
```

### Prepare Datasets

You can download the dataset _GossipCop_ from [ENDEF, SIGIR 2023](https://github.com/ICTMCG/ENDEF-SIGIR2022), and PolitiFact and Snopes from https://www.mpi-inf.mpg.de/dl-cred-analysis/
, and then place them to the folder `./data`;

### Train

- Run the shell script:
```shell
python main.py --model_name t5our --dataset gossip 
```
where `--dataset` includes gossip, politifact, snopes; `--model_name` contains t5, t5emo, t5mdfend, t5our 
(_t5ours_ represents our proposed DM-INTER model).


- Check log files in `./log`, and we prepare an automatic tool `read_results.py` to convert log files to an excel table. 

### Citation

- ArXiv version

```
@article{wang2024why,
  author       = {Bing Wang and
                  Ximing Li and
                  Changchun Li and
                  Bo Fu and
                  Songwen Pei and
                  Shengsheng Wang},
  title        = {Why Misinformation is Created? Detecting them by Integrating Intent
                  Features},
  journal      = {CoRR},
  volume       = {abs/2407.19196},
  year         = {2024},
}
```
