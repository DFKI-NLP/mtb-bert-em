from typing import Any, List, Dict, Callable
from logging import getLogger
from tqdm import tqdm
import pickle

import torch
from torch import nn
from torch.optim import Adam
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix


logger = getLogger(__name__)


def train_and_eval(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    eval_loader: torch.utils.data.DataLoader,
    label_to_id: Dict[Any, int],
    batch_processor: Callable,
    num_epochs: int = 5,
    lr: float = 3.0e-5,
    device: torch.device = torch.device("cpu"),
) -> float:
    # set loss function, optimizer and model device
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(params=model.parameters(), lr=lr)
    model.to(device)

    with tqdm(total=num_epochs * len(train_loader)) as pbar:
        for epoch in range(num_epochs):

            model.train()
            loss_epoch, labels_list, preds_list = 0, [], []
            for batch in train_loader:
                tokenized, cues = batch_processor(batch)
                tokenized.pop("offset_mapping")
                tokenized = {k: v.to(device) for k, v in tokenized.items()}
                labels = torch.tensor(batch["relation_id"], device=device)

                outputs = model(x=tokenized, cues=cues)
                loss = criterion(outputs, labels)
                loss_epoch += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(1)

            logger.info(
                "Epoch [{}/{}], Training Loss: {:.4f}.".format(
                    epoch + 1, num_epochs, loss_epoch
                )
            )

            with torch.no_grad():
                model.eval()
                labels_list, preds_list = [], []
                for batch in tqdm(eval_loader):
                    tokenized, cues = batch_processor(batch)
                    tokenized.pop("offset_mapping")
                    tokenized = {k: v.to(device) for k, v in tokenized.items()}

                    outputs = model(x=tokenized, cues=cues)
                    preds = torch.argmax(outputs, dim=1).detach().cpu().numpy().tolist()
                    preds_list.extend(preds)
                    labels_list.extend(batch["relation_id"])

            id_to_label = {v: k for k, v in label_to_id.items()}
            labels_list = [id_to_label[i] for i in labels_list]
            preds_list = [id_to_label[i] for i in preds_list]

            positive_labels = [
                label_name
                for label_name in label_to_id.keys()
                if label_name not in ["no_relation", "Other"]
            ]

            eval_f1 = f1_score(
                labels_list,
                preds_list,
                labels=positive_labels,
                average="micro",
            )
            logger.info("Micro-F1 score of evaluation: {:.4f}.".format(eval_f1))

            # save classification report
            cls_report = classification_report(
                labels_list,
                preds_list,
                labels=positive_labels,
                digits=4,
            )
            with open("classification_report.txt", "a") as f:
                f.write(cls_report)

            # save confusion matrix
            np.savetxt(
                "confusion_matrix.txt",
                confusion_matrix(labels_list, preds_list, labels=positive_labels),
                fmt="%i",
                delimiter=",",
            )

        # save labels and preds
        with open("labels", "wb") as fp:
            pickle.dump(labels_list, fp)
        with open("preds", "wb") as fp:
            pickle.dump(preds_list, fp)

    return eval_f1
