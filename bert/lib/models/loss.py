def compute_ner_loss(model, ner_res, tag):
    # reduction="mean"=the output will be averaged over batches.
    # sum or mean
    return -model.module.crf(ner_res, tag, mask=(tag != 0), reduction="sum")
