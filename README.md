# doom_rl
Analisi e comparazione di tre differenti algoritmi di Reinforcement Learning (PPO, DQN e A2C) nel gioco di Doom su tre ambienti diversi forniti dalla libreria VizDoom (DeadlyCorridor, DefendTheCenter, HealthGathering) e studio sull'ottimizzazione degli iperparametri per i modelli migliori

Tutti i modelli sviluppati sono disponibili al seguente [link](https://drive.google.com/drive/folders/1corM8rfArfy-ZTTIRUcE_bKYuVFaOvNK?usp=sharing), una volta scaricata la cartella "final_models" basterà copiarla nella root del progetto.

## Prerequisiti
Per l'esecuzione degli script in questo progetto e' necessario creare un conda environment col seguente comando:
```
conda env create -f environment.yml
```
per poi attivarlo con:
```
conda activate vizdoom
```
<br>

Inoltre per poter replicare la reward shaping di questo progetto e' necessario importare gli environment custom all'interno della libreria di VizDoom, questo accedendo al file \_\_init\_\_.py all'interno del pacchetto vizdoom (Per semplicità basta andare nello script train.py in un qualunque IDE e selezionare l'import di "vizdoom.gymnasium_wrapper" e premere F12).
Una volta nel file \_\_init\_\_.py corretto bisogna aggiungere i seguenti snippets al codice:
```
register(
    id="VizdoomCorridor-custom-v0",
    entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_file": "$BASEPATH/doom_rl/scenarios/deadly_corridor.cfg"},
)

register(
    id="VizdoomDefendCenter-custom-v0",
    entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_file": "$BASEPATH/doom_rl/scenarios/defend_the_center.cfg"},
)

register(
    id="VizdoomHealthGathering-custom-v0",
    entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_file": "$BASEPATH/doom_rl/scenarios/health_gathering.cfg"},
)
```
Sostituendo $BASEPATH con il proprio path.
<br><br>

## Replicazione
I seguenti script sono disponibili per la visualizzazione e la replicazione degli esperimenti:<br><br>

```
python evaluate.py --env DeadlyCorridor --model PPO --model_number_id 38 --eval_episodes 10
```
Per poter visualizzare uno dei modelli sviluppati in azione ed eseguire una valutazione su 
piu' episodi

<details>
<summary>Parametri per evaluate.py</summary>

- **--help**  
  Mostra una guida sui parametri.

- **--env** \
  Nome dell'ambiente di gioco (DeadlyCorridor, DefendTheCenter, HealthGathering).

- **--model**  
  Modello da utilizzare (PPO, DQN, A2C).

- **--model_number_id** \
  ID contenuto nel nome del modello allenato all'interno di final_models.

- **--eval_episodes** \
  Specifica il numero di episodi di valutazione da eseguire.

</details><br><br>


```
python train.py --env DeadlyCorridor --model PPO
```
Per poter replicare l'allenamento di uno dei modelli sviluppati

<details>
<summary>Parametri per train.py</summary>

- **--help**  
  Mostra una guida sui parametri.

- **--env** \
  Nome dell'ambiente di gioco (DeadlyCorridor, DefendTheCenter, HealthGathering).

- **--model**  
  Modello da utilizzare (PPO, DQN, A2C).

- **--use_best_params** \
  Specifica se allenare il modello usando i parametri di default o i parametri ottimizzati trovati.

- **--use_wandb** \
  Specifica se usare o meno il tracking dell'esperimento sfruttando il servizio di wandb (E' prima necessario inizializzare l'installazione di wandb come specificato al seguente [link](https://docs.wandb.ai/quickstart/)).

</details><br><br>


```
python optimizeHyperparameters.py --env DeadlyCorridor --model PPO --n_trials 50
```
Per poter replicare l'ottimizzazione degli iperparametri di uno dei modelli sviluppati

<details>
<summary>Parametri per optimizeHyperparameters.py</summary>

- **--help**  
  Mostra una guida sui parametri.

- **--env** \
  Nome dell'ambiente di gioco (DeadlyCorridor, DefendTheCenter, HealthGathering).

- **--model**  
  Modello da utilizzare (PPO, DQN, A2C).

- **--n_trials** \
  Specifica il numero di trials per l'ottimizzazione da eseguire.
</details><br><br>
