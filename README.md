# Individual bounded rationality destabilizes cooperative dynamics in human–AI groups

> Individual bounded rationality destabilizes cooperative dynamics in human–AI groups <br>
> accepted by *xxxx* as a research article <br>
> Kazushi Tsutsui, Nobuaki Mizuguchi, Yuta Goto, Ryoji Onagawa, Fumihiro Kano, Kazutoshi Kudo, Tadao Isaka, Keisuke Fujii
>
> [[biorxiv]](https://www.biorxiv.org/content/10.64898/2025.11.30.691459v1)

## Experimental setup and models
<img width="800" alt="Image" src="https://github.com/user-attachments/assets/1463fad3-ea2b-43aa-acb1-6619bb821d93" /> <br>
 (left) Triangular passing task. Three attackers attempted to maintain possession against a single defender, while the defender attempted to intercept. 
(center) Attacker model. The attacking agent followed heuristic rules such as passing to the wider angle and maintaining viable passing lanes for teammates. 
(right) Defender model. The defender was trained using deep reinforcement learning to intercept passes.

## Examples
<img width="600" alt="Image" src="https://github.com/user-attachments/assets/51cd5cad-1efc-49a6-a8c5-a0de9841a884" />

The videos are examples for each experimental condition. 
Each row varies the ball speed: slow (bottom), medium (middle), fast (top). 
Each column varies the defender speed: slow (left), medium (center), fast (right).

## Setup
- This repository was tested with Python 3.8
- To set up the environment, please run the following command: <br>
```pip install -r requirements.txt```

## Training
- To train the defensive agent (reinforcement learning model), please run ```train_defender.ipynb``` located in the ```train``` directory.

## Experiments
- To run the human experiments, execute ```exp_human.ipynb``` in the ```experiments``` directory. <br>
  Note: Running the human experiment requires a control device (e.g., Xbox One controller) connected to your computer.
- To run the agent experiments (computational simulations), execute ```exp_agent.ipynb``` in the same directory.  

- In both cases, you must first download the defensive model file ```model/defender.pth``` and place it at the top level of this repository.
- Alternatively, you may train the model yourself using the training notebook.

## Data availability
- The data and model are available in the following figshare repository. These data and models can be used to replicate the figures in the article in the ```notebooks``` directory.
```bash
https://doi.org/10.6084/m9.figshare.30648164
```

## Author
Kazushi Tsutsui ([@TsutsuiKazushi](https://github.com/TsutsuiKazushi)) <br>
E-mail: ```k.tsutsui6<at>gmail.com```
