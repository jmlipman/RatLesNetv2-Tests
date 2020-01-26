import numpy as np
from skimage import measure
from scipy import ndimage
from lib.utils import border_np

class Metric:
    def __init__(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

    def all(self):
        """Compute all metrics. Useful for evaluation.
           It assumes that BatchSize is 1
        """
        dice = list(self.dice()[0])
        hausdorff, hausdorff_anisotropy = self.hausdorff_distance()[0]
        islands = self.islands()[0]
        sens, prec, spec, tp, tn, fp, fn = self.sensivity_precision()
        sens, prec, spec, tp, tn, fp, fn = sens[0], prec[0], spec[0], tp[0], tn[0], fp[0], fn[0]
        comp1, comp2, comp_inv1, comp_inv2 = self.compactness()[0]
        
        return [dice, islands, hausdorff, sens, prec, spec, tp, tn, fp, fn, hausdorff_anisotropy, comp1, comp2, comp_inv1, comp_inv2]

    def dice(self):
        """This function calculates the Dice coefficient.

           Args:
            `y_pred`: batch containing the predictions. BDWHC.
            `y_true`: batch containing the predictions. BDWHC.

           Returns:
            Dice coefficient. BC (B: batch, C: classes)
        """
        num_samples = self.y_pred.shape[0]
        num_classes = self.y_pred.shape[1]
        results = np.zeros((num_samples, num_classes))
        #y_pred = np.argmax(self.y_pred, axis=1)
        #y_true = np.argmax(self.y_true, axis=1)

        for i in range(num_samples):
            for c in range(num_classes):
                #a = y_pred[i] == c
                #b = y_true[i] == c
                a = 1.0*(self.y_pred[i,c] > 0.5)
                b = 1.0*(self.y_true[i,c] > 0.5)
                if np.sum(b) == 0: # If no lesion in the y_true
                    if np.sum(a) == 0: # No lesion predicted
                        result = 1.0
                    else:
                        result = (np.sum(b==0)-np.sum(a))*1.0 / np.sum(b==0)
                else: # Actual Dice
                    num = 2 * np.sum(a * b)
                    denom = np.sum(a) + np.sum(b)
                    result = num / denom
                results[i, c] = result
        return results

    def islands(self):
        """Returns the number of islands i.e. independently connected components.

           Args:
           `y`: output from the network, B2WHD
        """
        num_samples = self.y_pred.shape[0]
        results = np.zeros(num_samples)
        for i in range(num_samples):
            # NOTE: I will leave this like this but ideally I would count every channel.
            results[i] = np.max(measure.label(np.argmax(self.y_pred[i], axis=0)))
        return results

    def hausdorff_distance(self):
        """Hausdorff distance.
           From NiftyNet.
           2-classes only!
        """
        num_samples = self.y_pred.shape[0]
        results = np.zeros((num_samples, 2))
        for i in range(num_samples):
            y_pred = np.argmax(self.y_pred[i], axis=0)
            y_true = np.argmax(self.y_true[i], axis=0)
            ref_border_dist, seg_border_dist = self._border_distance(y_pred, y_true)
            results[i,0] = np.max([np.max(ref_border_dist), np.max(seg_border_dist)])

            # Increasing the number of slices, to 9 per slice.
            y_pred2 = np.zeros((18*9,256,256))
            y_true2 = np.zeros((18*9,256,256))
            for j in range(18):
                y_true2[j*9:(j+1)*9,:,:] = y_true[j:j+1,:,:]
                y_pred2[j*9:(j+1)*9,:,:] = y_pred[j:j+1,:,:]
            ref_border_dist2, seg_border_dist2 = self._border_distance(y_pred2, y_true2)
            results[i,1] = np.max([np.max(ref_border_dist2), np.max(seg_border_dist2)])

        return results

    def compactness(self):
        """surface^1.5 / volume
        """
        num_samples = self.y_pred.shape[0]
        results = np.zeros((num_samples, 4))
        for i in range(num_samples):
            pred = np.argmax(self.y_pred[i], axis=0)
            pred2 = np.zeros((18*9,256,256))
            for j in range(18):
                pred2[j*9:(j+1)*9,:,:] = pred[j:j+1,:,:]

            surface1 = np.sum(border_np(pred))
            volume1 = np.sum(pred)
            surface2 = np.sum(border_np(pred2))
            volume2 = np.sum(pred2)
            results[i] = [volume1/(surface1**1.5), volume2/(surface2**1.5), (surface1**1.5)/volume1, (surface2**1.5)/volume2]
        return results

    def sensivity_precision(self):
        """ Sensivity: Among the real lesion voxels, how many I can accurately predict.
            Precision: Among the predicted lesion voxels, how many I can accurately predict.
            Note this assumes only two classes!
        """

        num_samples = self.y_pred.shape[0]
        sensivity = np.zeros(num_samples)
        precision = np.zeros(num_samples)
        specificity = np.zeros(num_samples)
        TP = np.zeros(num_samples)
        TN = np.zeros(num_samples)
        FP = np.zeros(num_samples)
        FN = np.zeros(num_samples)

        for i in range(num_samples):

            y_pred = np.argmax(self.y_pred[i], axis=0)
            y_true = np.argmax(self.y_true[i], axis=0)


            TP[i] = int(np.sum(y_pred*y_true))
            TN[i] = int(np.sum((1-y_pred)*(1-y_true)))
            FP[i] = int(np.sum(y_pred*(1-y_true)))
            FN[i] = int(np.sum((1-y_pred)*y_true))

            sensivity[i] = TP[i] / (TP[i] + FN[i] + 1e-10)
            precision[i] = TP[i] / (TP[i] + FP[i] + 1e-10)
            specificity[i] = TN[i] / (TN[i] + FP[i] + 1e-10)

        return [sensivity, precision, specificity, TP, TN, FP, FN]

    def _border_distance(self, y_pred, y_true):
        """Distance between two borders.
           From NiftyNet.
           y_pred and y_true are WHD
        """
        border_seg = border_np(y_pred)
        border_ref = border_np(y_true)
        distance_ref = ndimage.distance_transform_edt(1 - border_ref)
        distance_seg = ndimage.distance_transform_edt(1 - border_seg)
        distance_border_seg = border_ref * distance_seg
        distance_border_ref = border_seg * distance_ref
        return distance_border_ref, distance_border_seg
