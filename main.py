import logging

import torch
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf

from mtb.data import TACREDDataset, SemEvalDataset
from mtb.model import MTBModel
from mtb.processor import BatchTokenizer, aggregate_batch
from mtb.train_eval import train_and_eval
from mtb.utils import resolve_relative_path, seed_everything


logger = logging.getLogger(__name__)


@hydra.main(config_name="config", config_path="configs")
def main(cfg: DictConfig) -> None:
    """
    Conducts evaluation given the configuration.
    Args:
        cfg: Hydra-format configuration given in a dict.
    """
    resolve_relative_path(cfg)
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)

    # prepare dataset: parse raw dataset and do some simple pre-processing such as
    # convert special tokens and insert entity markers
    entity_marker = True if cfg.variant in ["d", "e", "f"] else False
    if "tacred" in cfg.train_file.lower():
        train_dataset = TACREDDataset(cfg.train_file, entity_marker=entity_marker)
        eval_dataset = TACREDDataset(cfg.eval_file, entity_marker=entity_marker)
        layer_norm = False
    elif "semeval" in cfg.train_file.lower():
        train_dataset = SemEvalDataset(cfg.train_file, entity_marker=entity_marker)
        eval_dataset = SemEvalDataset(cfg.eval_file, entity_marker=entity_marker)
        layer_norm = True
    label_to_id = train_dataset.label_to_id

    # set dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        collate_fn=aggregate_batch,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=aggregate_batch,
    )

    # set a processor that tokenizes and aligns all the tokens in a batch
    batch_processor = BatchTokenizer(
        tokenizer_name_or_path=cfg.model,
        variant=cfg.variant,
        max_length=cfg.max_length,
    )
    vocab_size = len(batch_processor.tokenizer)

    # set model and device
    model = MTBModel(
        encoder_name_or_path=cfg.model,
        variant=cfg.variant,
        layer_norm=layer_norm,
        vocab_size=vocab_size,
        num_classes=len(label_to_id),
        dropout=cfg.dropout,
    )
    device = (
        torch.device("cuda", cfg.cuda_device)
        if cfg.cuda_device > -1
        else torch.device("cpu")
    )

    eval_result = train_and_eval(
        model,
        train_loader,
        eval_loader,
        label_to_id,
        batch_processor,
        num_epochs=cfg.num_epochs,
        lr=cfg.lr,
        device=device,
    )
    logger.info("Evaluation micro-F1: {:.4f}".format(eval_result))


if __name__ == "__main__":
    main()
