import matplotlib.pyplot as plt
from sklearn.metrics import  precision_recall_curve, RocCurveDisplay, auc
import numpy as np
import shap
import os

def get_class_weight(neg, pos):
    total = neg + pos
    weight_for_0 = (1 / neg) * (total) / 2.0
    weight_for_1 = (1 / pos) * (total) / 2.0
    class_weight = {0: weight_for_0, 1: weight_for_1}
    return class_weight

def calculate_specificity(true_pred, thresholds):
    # True Negative (TN)과 False Positive (FP) 초기화
    TN = 0
    FP = 0
    
    # 임계값(threshold)마다 TN과 FP를 계산
    for i in range(len(true_pred)):
        if true_pred[i] < thresholds:
            TN += 1
        else:
            FP += 1
    
    # Specificity 계산
    specificity = TN / (TN + FP)
    
    return specificity

def analyze(res, title="GBM", save_path= ",/good/", hospital_name="SNUH"):
    # ROC
    tprs = []
    aucs = []
    auprcs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(6, 6))

    
    # if hospital_name == "SNUH" :
    #     csv_file = pd.read_csv("SNUH_test_file.csv")
    # else : 
    #     csv_file = pd.read_csv("CNUH_test_file.csv")
    
    # if cfg.train_param.category_name == "sex" or cfg.train_param.category_name == "age" :
    #     if cfg.train_param.category_name == "sex" :
    #         category_name = "sex_bi"
    #     elif cfg.train_param.category_name == "age":
    #         category_name = "age_range"
        
    #     category = cfg.train_param.category
    #     csv_file = csv_file[csv_file[category_name] == category]


    for fold, (true, pred) in enumerate(res):
        viz = RocCurveDisplay.from_predictions(
            true,
            pred,
            name=f"ROC fold {fold+1}",
            alpha=0.3,
            lw=1,
            ax=ax,
            plot_chance_level=(fold == 4),
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        #print(interp_tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

        # AUPRC
        precision, recall, thresholds = precision_recall_curve(true, pred)
        auprcs.append(auc(recall, precision))
    

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=title,
    )
    ax.legend(loc="lower right")
    plt.savefig(save_path + f"/{hospital_name}.png")

    # statistics

    print("AUROC(std) ", mean_auc, "(", std_auc, ")")
    auprcs = np.array(auprcs)
    print("AUPRC(std) ", auprcs.mean(), "(", auprcs.std(), ")")

    best_fold = np.array(aucs).argmax()
    true, pred = res[best_fold]

    precision, recall, thresholds = precision_recall_curve(true, pred)
    numerator = 2 * recall * precision
    denom = recall + precision
    f1_scores = []
    
    f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
    max_f1_score = np.max(f1_scores)
    opt_idx = np.where(f1_scores==max_f1_score)[0][0]

    print(thresholds[opt_idx])
    print("Accuracy ", np.round((true == (pred > thresholds[opt_idx])).mean(),4))
    print("Precision ", np.round(precision[opt_idx],4))
    print("Recall ", np.round(recall[opt_idx],4))
    print("Specificity ", np.round(calculate_specificity(pred, thresholds[opt_idx]),4))
    print("F1 score ", np.round(max_f1_score,4))
    print("AUROC ", np.round(aucs[best_fold],4))
    print("AUPRC ", np.round(auprcs[best_fold],4))
    
    return best_fold
    

def visualization_gbm(history, paths):
    
    plt.subplot(2,1,1)
    plt.ylim([0, 1])
    plt.plot(history["logloss"])
    # plt.plot(history["val_loss"])
    plt.title("logloss")


    last_accuracy = history["auc"][-1]
    # last_val_accuracy = history["val_auc"][-1]
    
    plt.subplot(2,1,2)
    plt.plot(history["auc"])
    # plt.plot(history["val_auc"])
    plt.text(len(history["auc"]) - 1, last_accuracy, f'{last_accuracy:.4f}', color='blue', ha='center', va='bottom')
    # plt.text(len(history["val_auc"]) - 1, last_val_accuracy, f'{last_val_accuracy:.4f}', color='orange', ha='center', va='bottom')
    plt.title("auc")
    
    
    plt.tight_layout()
    plt.savefig(paths)
    plt.clf()

def visualization(history, paths):
    
    plt.subplot(3,1,1)
    plt.ylim([0, 1])
    plt.plot(history["loss"])
    plt.plot(history["val_loss"])
    plt.title("loss")


    last_accuracy = history["auc"][-1]
    last_val_accuracy = history["val_auc"][-1]
    
    plt.subplot(3,1,2)
    plt.plot(history["auc"])
    plt.plot(history["val_auc"])
    plt.text(len(history["auc"]) - 1, last_accuracy, f'{last_accuracy:.4f}', color='blue', ha='center', va='bottom')
    plt.text(len(history["val_auc"]) - 1, last_val_accuracy, f'{last_val_accuracy:.4f}', color='orange', ha='center', va='bottom')
    plt.title("auc")

    last_accuracy = history["prc"][-1]
    last_val_accuracy = history["val_prc"][-1]

    plt.subplot(3,1,3)
    plt.plot(history["prc"])
    plt.plot(history["val_prc"])
    plt.text(len(history["prc"]) - 1, last_accuracy, f'{last_accuracy:.4f}', color='blue', ha='center', va='bottom')
    plt.text(len(history["val_prc"]) - 1, last_val_accuracy, f'{last_val_accuracy:.4f}', color='orange', ha='center', va='bottom')
    plt.title("prc")    
    
    plt.tight_layout()
    plt.savefig(paths)
    plt.clf()

def encode_sex(sex):
    if sex.lower() == 'm':
        return 0
    elif sex.lower() == 'f':
        return 1
    else:
        raise ValueError("Invalid input. Please provide 'm' or 'f'.")

def analyze_shap(model, x_test, input_shape, path) :
    plt.clf()
    features_name = ["spo2", "etco2", "fio2", "tv", "pip", "mv", "sex", "age", "weight", "height"]
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test)
    shap_values_six_variables = np.mean(shap_values[:, :-4].reshape(-1,input_shape[0],input_shape[1]), axis=1)
    shap_values_demography = shap_values[:, -4:] 
    shap_values_concate = np.concatenate([shap_values_six_variables, shap_values_demography], axis=1)


    x_test_six_variables = np.mean(x_test[:, :-4].reshape(-1,input_shape[0],input_shape[1]), axis=1) 
    x_test_demography = x_test[:, -4:] 
    x_test_concate = np.concatenate([x_test_six_variables, x_test_demography], axis=1)
    
    shap.summary_plot(shap_values= shap_values_concate, features= x_test_concate, feature_names=features_name, show=False)
    plt.savefig(path, dpi=300)
    plt.clf()

    shap.initjs()
    top_inds = np.argsort(-np.sum(np.abs(shap_values_concate), 0))

    # make SHAP plots of the three most important features
    plt.figure(figsize=(8,6))
    
    if not os.path.exists(path.split("/")[0]+"/SNUH/") : 
        os.makedirs(path.split("/")[0]+"/SNUH/")
    if not os.path.exists(path.split("/")[0]+"/CNUH/") : 
        os.makedirs(path.split("/")[0]+"/CNUH/")
    
    for i in range(10):
        shap.dependence_plot(top_inds[i], shap_values_concate, x_test_concate, feature_names=features_name, show=False)
        

        if "SNUH" in path.split("/")[-1] :
            plt.tight_layout()
            plt.savefig(path.split("/")[0] + f"/SNUH\\{i}.png", dpi=300)
        else :
            plt.tight_layout()
            plt.savefig(path.split("/")[0] + f"/CNUH\\{i}.png", dpi=300)

        plt.clf()

def normalize(arr):
    arr = arr[:, :180].reshape(-1, 30, 6)
    return (arr - arr.mean((0, 1), keepdims=True)) / arr.std((0, 1), keepdims=True)


