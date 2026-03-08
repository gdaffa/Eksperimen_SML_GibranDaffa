# Eksperimen Sistem Machine Learning (SML)

> [!NOTE]
> This repo is part of the `Membangun Sistem Machine Learning` final project
> from Dicoding, referring to
> [this repo](https://github.com/gdaffa/dicoding-machine-learning/tree/main/5.%20Membangun%20Sistem%20Machine%20Learning/Proyek%20Akhir).

A sub-project for Dicoding final project to perform a **continous integration**
by train a **machine learning model** and deploy the result to **Docker**.

The project instruction ask me to create 2 separate repo,
`Eksperimen_SML_<username>` (mandatory) and `Workflow-CI` (not mandatory). I
decided to combine both into `Eksperimen_SML_GibranDaffa` to make the continous
integration looks clear and seamless.

Those repo has a different task:
1. `Eksperimen_SML_GibranDaffa`: Turning a raw data into a preprocessed data.
2. `Workflow-CI`: Train a model with preprocessed data that will be deployed to
   Docker and Github.

The CI script also has 2 different jobs:
1. `preprocess` which belongs to `Eksperimen_SML_GibranDaffa`.
2. `integrate` which belongs to `Workflow-CI`.

For clarity, each folder is representing a different repo:
1. **Eksperimen_SML_GibranDaffa**: `preprocessing`, `joblibs`, `transformer`.
2. **Workflow-CI**: `train`, `mlruns`.
3. **Both**: `dataset`.

The model was successfully deployed to the Docker hub, which can be found on
[gibrandaffa/submission-mlflow-docker](https://hub.docker.com/r/gibrandaffa/submission-mlflow-docker).

## License

The [raw dataset](https://www.openml.org/search?type=data&sort=runs&id=42225)
licensed under CC0 license, while this repo licensed under MIT license.
