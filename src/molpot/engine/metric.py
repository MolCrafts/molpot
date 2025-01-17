

def get_tensor(pred_label, label):
    def _extract_metric_tensor(outputs):
        return (outputs["predicts"][pred_label], outputs["labels"][label])
    return _extract_metric_tensor