def sMAPE(prediction, ground_truth, horizon):
    add = 0
    for pred, gt in zip(prediction, ground_truth):
        add += 2*(abs(gt - pred)/(abs(gt)+abs(pred)))
    return add/horizon