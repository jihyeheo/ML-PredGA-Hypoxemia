# ML-PredGA-Hypoxemia
ML based hypoxemia prediction in general anesthesia patients

Sujin Baek*, Jung-Bin Park*, **Jihye heo**, Kyungsang Kim, Donghyeon Baek, Chahyun Oh, Hyung-Chul Lee, Dongheon Lee and Boohwi Hong



### Setup
```bash
conda env create --file environment.yml
conda activate hypoxemia
```

* train
```bash
python main.py --phase train --model <custom model>
```

* test <br>
Once the test is completed, the evaluation metrics for each of the 5 folds, the AUROC graph, and the SHAP values (in the case of XGBoost) can be obtained.
```bash
python main.py --phase test --model <custom model> --checkpoint-dir <your weight file path>
```
