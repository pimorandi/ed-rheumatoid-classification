import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as mtr

from .utils import write_json

class Evaluator():
    def __init__(self):
        pass

    def eval_roc(self, y_true, score):
        fpr, tpr, thr = mtr.roc_curve(y_true, score)
        auc = mtr.roc_auc_score(y_true, score)
        return {'fpr': fpr, 'tpr': tpr, 'thr': thr, 'auc': auc}
    
    def get_best_thr(self, roc):
        df_roc = pd.DataFrame(roc)
        df_roc['tf'] = df_roc.apply(lambda r: r.tpr - r.fpr, axis=1)
        best_thr = df_roc.sort_values(by='tf', ascending=False).thr.values[0]
        return best_thr

    def plot_roc(self, y_true, score):
        roc = self.eval_roc(y_true, score)

        f, ax = plt.subplots()
        _ = ax.plot(roc['fpr'], roc['tpr'], label='roc curve')
        _ = ax.plot([0,1], [0,1], '--', c='gray', label='No skill')
        _ = ax.set_xlabel('FPR')
        _ = ax.set_ylabel('TPR')
        auc = roc['auc']
        _ = ax.fill_between(roc['fpr'], roc['tpr'], alpha=0.3, label=f"AUC: {auc:.3f}")
        _ = ax.grid(alpha=0.4)
        _ = ax.legend()
        return f, ax
    
    def plot_probability_distribution(self, df):
        f, ax = plt.subplots()

        x = np.linspace(0,1,50)
        _ = ax.hist(df.query("y_true==0").score, bins=x, alpha=0.4, label='Class 0')
        _ = ax.hist(df.query("y_true==1").score, bins=x, alpha=0.4, label='Class 1')
        _ = ax.legend()
        _ = ax.grid(alpha=0.4)
        _ = ax.set_xlabel('Model score')
        _ = ax.set_ylabel('[#]')
        _ = ax.set_title('Output probability distribution')
        return f, ax
    
    def get_scores(self, y_true, y_pred):
        scores = {
            'precision': mtr.precision_score(y_true, y_pred, average=None).tolist(),
            'recall': mtr.recall_score(y_true, y_pred, average=None).tolist(),
            'f1_score': mtr.f1_score(y_true, y_pred, average=None).tolist(),
        }
        return scores
    
    def sigmoid(self, x):
        return np.exp(-np.logaddexp(0, -x))
    
    def eval(self, trainer, dataset, best_thr=None):
        pred_output = trainer.predict(dataset)
        scores = [self.sigmoid(x) for x in pred_output.predictions.squeeze()]
        preds = pd.DataFrame({'score': scores, 'anon_id': dataset['anon_id']})
        preds['y_true'] = dataset['label']

        roc = self.eval_roc(preds.y_true, preds.score)
        if best_thr is None:
            best_thr = self.get_best_thr(roc)
        preds['y_pred'] = preds.score.apply(lambda x: 1 if x > best_thr else 0)
        scores = self.get_scores(preds.y_true, preds.y_pred)
        scores['roc_auc'] = roc['auc']
        return {'predictions': preds, 'roc': roc, 'scores': scores}, best_thr
    
    def run(self, trainer, train_dataset, test_dataset, save_path=None):
        train_results, best_thr = self.eval(trainer, train_dataset)
        test_results, _ = self.eval(trainer, test_dataset, best_thr=best_thr)
        return train_results, test_results, best_thr
    
    def save_results(self, save_obj, path, prefix='train'):
        save_obj['predictions'].to_csv(path / f'{prefix}_predictions.csv')
        write_json(save_obj['scores'], path / f'{prefix}_scores.json')
        f, ax = self.plot_roc(save_obj['predictions'].y_true, save_obj['predictions'].score)
        f.savefig(path / f'{prefix}_roc.jpg')
        plt.close()
        f, ax = self.plot_probability_distribution(save_obj['predictions'])
        f.savefig(path / f'{prefix}_prob_distribution.jpg')
        plt.close()
        return None