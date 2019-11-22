import numpy as np


class runningScore(object):
    def __init__(self, n_classes, n_val):
        self.n_classes = n_classes
        self.n_val = n_val
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.dsc_cls = np.zeros((n_classes, n_val))
        self.acc = np.zeros(n_val)


    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds, i_val):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)
        # dsc = (2 * np.diag(self.confusion_matrix)) / (self.confusion_matrix.sum(axis=1) + self.confusion_matrix.sum(axis=0))
        # self.acc[i_val] = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        # for i in range(self.n_classes):
        #     self.dsc_cls[i][i_val] = dsc[i]

    def get_list(self):
        return self.acc, self.dsc_cls

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - Dice (DSC): dice = 2*TP/(TP+FP+TP+FN)
            - Pixel Accuracy (PA)
		"""

        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        # dice = (2 * iu) / (iu + 1)
        dice = (2 * np.diag(hist)) / (hist.sum(axis=1) + hist.sum(axis=0))
        mean_dice = np.nanmean(dice)
        mean_iu = np.nanmean(iu)

        cls_dice = dict(zip(range(self.n_classes), dice))

        # cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "PA: \t": acc,
                "JS : \t": mean_iu,
                "Dice : \t": mean_dice,
            },
            cls_dice,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count