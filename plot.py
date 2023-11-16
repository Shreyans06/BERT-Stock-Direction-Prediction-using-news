import torch
import matplotlib.pyplot as plt
import numpy as np

load_metrics = torch.load("./metrics/amazon_news_daily_summarized_spacy_ProsusAI_finbert_spacy_text_128_16_1e-05/1.pt")

epochs = [i + 1 for i in range(len(load_metrics['val_accuracy']))]

plt.plot(  epochs , load_metrics['val_accuracy'] , label = 'Validation Accuracy')
plt.plot(  epochs , load_metrics['train_accuracy'] , label = 'Training Accuracy')

plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.xticks(np.arange(1, len(epochs) + 1, 2))

# Display the plot
plt.legend(loc='best')
plt.savefig(f"./plots/val_vs_train_acc.png")

plt.clf()

plt.plot(epochs , load_metrics['val_loss'] , label = 'Validation Loss')
plt.plot(epochs , load_metrics['train_loss'] , label = 'Training Loss')

plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.xticks(np.arange(1, len(epochs) + 1, 2))

# Display the plot
plt.legend(loc='best')
plt.savefig(f"./plots/val_vs_train_loss.png")
plt.show()

# print(load_metrics)

load_report = torch.load("./plots/amazon_news_daily_summarized_spacy_ProsusAI_finbert_spacy_text_128_16_1e-05_report.pt")
# print(load_report)

def report_average(reports):
    mean_dict = dict()
    for label in reports[0].keys():
        dictionary = dict()

        if label in 'accuracy':
            mean_dict[label] = sum(d[label] for d in reports) / len(reports)
            continue

        for key in reports[0][label].keys():
            dictionary[key] = sum(d[label][key] for d in reports) / len(reports)
        mean_dict[label] = dictionary

    return mean_dict

def classification_report(data_dict):
    """Build a text report showing the main classification metrics.
    Read more in the :ref:`User Guide <classification_report>`.
    Parameters
    ----------
    report : string
        Text summary of the precision, recall, F1 score for each class.
        Dictionary returned if output_dict is True. Dictionary has the
        following structure::
            {'label 1': {'precision':0.5,
                         'recall':1.0,
                         'f1-score':0.67,
                         'support':1},
             'label 2': { ... },
              ...
            }
        The reported averages include macro average (averaging the unweighted
        mean per label), weighted average (averaging the support-weighted mean
        per label), and sample average (only for multilabel classification).
        Micro average (averaging the total true positives, false negatives and
        false positives) is only shown for multi-label or multi-class
        with a subset of classes, because it corresponds to accuracy otherwise.
        See also :func:`precision_recall_fscore_support` for more details
        on averages.
        Note that in binary classification, recall of the positive class
        is also known as "sensitivity"; recall of the negative class is
        "specificity".
    """

    non_label_keys = ["accuracy", "macro avg", "weighted avg"]
    y_type = "binary"
    digits = 2

    target_names = [
        "%s" % key for key in data_dict.keys() if key not in non_label_keys
    ]

    # labelled micro average
    micro_is_accuracy = (y_type == "multiclass" or y_type == "binary")

    headers = ["precision", "recall", "f1-score", "support"]
    p = [data_dict[l][headers[0]] for l in target_names]
    r = [data_dict[l][headers[1]] for l in target_names]
    f1 = [data_dict[l][headers[2]] for l in target_names]
    s = [data_dict[l][headers[3]] for l in target_names]

    rows = zip(target_names, p, r, f1, s)

    if y_type.startswith("multilabel"):
        average_options = ("micro", "macro", "weighted", "samples")
    else:
        average_options = ("micro", "macro", "weighted")

    longest_last_line_heading = "weighted avg"
    name_width = max(len(cn) for cn in target_names)
    width = max(name_width, len(longest_last_line_heading), digits)
    head_fmt = "{:>{width}s} " + " {:>9}" * len(headers)
    report = head_fmt.format("", *headers, width=width)
    report += "\n\n"
    row_fmt = "{:>{width}s} " + " {:>9.{digits}f}" * 3 + " {:>9}\n"
    for row in rows:
        report += row_fmt.format(*row, width=width, digits=digits)
    report += "\n"

    # compute all applicable averages
    for average in average_options:
        if average.startswith("micro") and micro_is_accuracy:
            line_heading = "accuracy"
        else:
            line_heading = average + " avg"

        if line_heading == "accuracy":
            avg = [data_dict[line_heading], sum(s)]
            row_fmt_accuracy = "{:>{width}s} " + \
                    " {:>9.{digits}}" * 2 + " {:>9.{digits}f}" + \
                    " {:>9}\n"
            report += row_fmt_accuracy.format(line_heading, "", "",
                                              *avg, width=width,
                                              digits=digits)
        else:
            avg = list(data_dict[line_heading].values())
            report += row_fmt.format(line_heading, *avg,
                                     width=width, digits=digits)
    return report

reports = list(load_report.values())
avg_report = report_average(reports)
print(classification_report(avg_report))