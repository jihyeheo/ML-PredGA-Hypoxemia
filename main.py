import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from model import *
from utils import *
import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import KFold
import shutil
from easydict import EasyDict as edict
from sklearn.preprocessing import MinMaxScaler

# Parameters
with open("config.yaml", "r") as f:
    cfg = edict(yaml.safe_load(f))
RANDOM_SEED = 0
init_lr = cfg.train_param.init_lr
num_epochs = cfg.train_param.num_epochs
input_shape = cfg.train_param.input_shape
batch_size = cfg.train_param.batch_size
window_size = cfg.train_param.window_size
sample_interval = cfg.train_param.sample_interval
data_dir = "./Create_Data/SNUH_train/dependent/"

assert window_size == 30 or sample_interval == 1

## 변경
cate_sex_or_age = cfg.train_param.category_name
cate = cfg.train_param.category


# Seed
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# optimizer/scheduler
optimizer = tf.keras.optimizers.Adam(learning_rate=init_lr)  # init_lr)  # , global_clipnorm=0.1)

# args
parser = argparse.ArgumentParser()
parser.add_argument("--phase")
parser.add_argument("--model", default="gbm", choices=["gbm", "lstm", "transformer", "inception", "hivecote", "lightgbm"])
parser.add_argument("--checkpoint-dir", type=str)
args = parser.parse_args()
save_path = Path("weights_test/" + datetime.now().strftime("%m%d%H%M%S")+args.model)#+cate_sex_or_age+str(cate))  # Date to the desired string format
if not args.checkpoint_dir:
    save_path.mkdir(parents=True, exist_ok=True)  # Create folder
    shutil.copy("config.yaml", save_path / "config.yaml")

# Metrics
METRICS = [
    keras.metrics.BinaryCrossentropy(name="cross entropy"),
    keras.metrics.AUC(name="auc"),
    keras.metrics.AUC(name="prc", curve="PR"),
]

if args.phase == "train":
    skf = KFold(n_splits=5)
    xy = np.load(f"./Dataset/modify_dataset/SNUH_train.npy")
    #xy = np.load(f"./{cate_sex_or_age}/SNUH_gbm_train_{cate_sex_or_age}_{cate}.npy")
    x = xy[:, :-1]
    x1 = x[:, :180]
    x2 = x[:, 180:]
    # scaler = MinMaxScaler()
    # x1 = scaler.fit_transform(x1)
    x = np.concatenate([x1, x2], axis=1)
    

    y = xy[:, -1]
    for idx, (train_index, val_index) in enumerate(skf.split(xy)):
        # Define model
        model = {
            "gbm": build_gbm(),
            "lstm": build_lstm(),
            "transformer": build_transformer(),
            "inception": Classifier_INCEPTION().build_model(),
        }[args.model]
        

        x1_train, x2_train = x[train_index, :180].reshape(-1,30,6)[:,-window_size:], x[train_index, 180:]#.reshape((-1, 30, 6))
        x1_train = x1_train[:,::sample_interval].reshape(x1_train.shape[0], -1) # 10,6

        y_train = y[train_index]
        x_train = np.concatenate((x1_train, x2_train), axis=1)

        x1_val, x2_val = x[val_index, :180].reshape(-1,30,6)[:,-window_size:], x[val_index, 180:]
        x1_val = x1_val[:,::sample_interval].reshape(x1_val.shape[0], -1)

        y_val = y[val_index]

        x_val = np.concatenate((x1_val, x2_val), axis=1)

        if args.model in ["gbm"]:
            model.fit(x_train, y_train, eval_set=[(x_val, y_val)])
            model.save_model(save_path / f"{idx}_model")
        else:
            model.compile(optimizer=optimizer, loss=keras.losses.BinaryCrossentropy(), metrics=METRICS)
            loss_callback = ModelCheckpoint(save_path / f"{idx}_model", monitor="val_auc", save_best_only=True, mode="max", verbose=1)

            history = model.fit(
                x=[x1_train.reshape(-1,30,6), x2_train],
                y=y_train,
                validation_data=([x1_val.reshape(-1,30,6), x2_val], y_val),
                batch_size=batch_size,
                epochs=num_epochs,
                class_weight=get_class_weight(5209787, 61223),
                callbacks=[loss_callback],
            ).history

            model.save(save_path / f"{idx}_model_last")
            with open(save_path / "history.json", "w") as f:
                json.dump(history, f)
            visualization(history, save_path / f"{idx}.png")
        

elif args.phase == "test":
    for hospital_name in ["SNUH", "CNUH"]:
        xy = np.load(f"./Dataset/modify_dataset/{hospital_name}_test.npy")

        #xy = np.load(f"./{cate_sex_or_age}/{hospital_name}_gbm_test_{cate_sex_or_age}_{cate}.npy")

        x = xy[:, :-1]
        y_true = xy[:, -1]
        x1 = x[:, :180]
        x2 = x[:, 180:]
        x1 = x1.reshape(-1,30,6)[:,-window_size:]
        x1 = x1[:,::sample_interval].reshape(x1.shape[0], -1) # 10,6
        #scaler = MinMaxScaler()
        #x1 = scaler.fit_transform(x1)
        x_test= np.concatenate((x1, x2), axis=1)

        res = []
        for idx in range(5):

            if args.model == "gbm":
                model = build_gbm()
                model.load_model(args.checkpoint_dir + f"/{idx}_model")
                y_proba = model.predict_proba(x_test)[:, 1]
                res.append((y_true, y_proba))
                
            else:
                model = load_model(args.checkpoint_dir + f"{idx}_model")
                model.compile(optimizer=optimizer, loss=keras.losses.BinaryCrossentropy(), metrics=METRICS)
                print(x2.shape)
                y_proba = model.predict([x1.reshape(-1,30,6), x2])[:, 0]
                res.append((y_true, y_proba))
        best_fold = analyze(res, title= args.model, save_path=args.checkpoint_dir, hospital_name=hospital_name)
        plt.clf()

        # 가장 좋을때의 auroc 값 출력
        true, pred = res[best_fold]
        #np.save(args.checkpoint_dir+ f"{hospital_name}_{args.model}_true.npy", np.array(true))
        #np.save(args.checkpoint_dir+ f"{hospital_name}_{args.model}_pred.npy", np.array(pred))
        
        viz = RocCurveDisplay.from_predictions(
            true,
            pred,
        )
        interp_tpr = np.interp(np.linspace(0, 1, 100), viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        #np.save(args.checkpoint_dir + f"{hospital_name}_auroc_value.npy", np.array(interp_tpr))
        #plt.clf()
        precision, recall, _ = precision_recall_curve(true, pred)
        interp_recall = np.linspace(0, 1, 100)
        interp_precision = np.interp(interp_recall, recall[::-1], precision[::-1])
        #print("저장 왜 안돼?", interp_precision)
        #np.save(args.checkpoint_dir + f"{hospital_name}_auprc_value.npy", np.array(interp_precision))
        
        
        #save shap plot
        if args.model == "gbm":
            model.load_model(args.checkpoint_dir + f"/{best_fold}_model")
            analyze_shap(model, x_test, cfg.train_param.input_shape, args.checkpoint_dir + f"{hospital_name}_shap_analysis.png")
        else : 
            model = build_lstm()
            model = load_model(args.checkpoint_dir + f"{best_fold}_model")
            model.compile(optimizer=optimizer, loss=keras.losses.BinaryCrossentropy(), metrics=METRICS)
            #analyze_shape_2d(model, [x1.reshape(-1,30,6), x2])
else:
    raise KeyError()
