from logging import getLogger
from multiprocessing import pool
from typing import List, Dict, Any, Tuple, Optional

import torch
from torch import nn
from transformers import AutoModel


logger = getLogger(__name__)


class MTBModel(nn.Module):
    def __init__(
        self,
        encoder_name_or_path: str = "bert-base-cased",
        variant: str = "b",
        layer_norm: bool = False,
        vocab_size: int = 29000,
        num_classes: int = 42,
        dropout: float = 0,
    ):
        super().__init__()
        self.variant = variant
        self.vocab_size = vocab_size

        self.encoder = AutoModel.from_pretrained(encoder_name_or_path)
        self.encoder.resize_token_embeddings(self.vocab_size)

        if self.variant == "c":
            self.encoder.config.type_vocab_size = 3
            token_type_embed = nn.Embedding(
                self.encoder.config.type_vocab_size, self.encoder.config.hidden_size
            )
            token_type_embed.weight.data.uniform_(-1, 1)
            self.encoder.embeddings.token_type_embeddings = token_type_embed

        self.hidden_size = self.encoder.config.hidden_size
        self.in_features = (
            self.hidden_size if self.variant in ["a", "d"] else 2 * self.hidden_size
        )

        if layer_norm == False:
            self.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features=self.in_features, out_features=num_classes),
            )
        else:
            self.fc = nn.Sequential(
                nn.LayerNorm(self.in_features),
                nn.Dropout(p=dropout),
                nn.Linear(in_features=self.in_features, out_features=num_classes),
            )

    def forward(
        self, x: Dict[str, Any], cues: Optional[Tuple[torch.Tensor]] = None
    ) -> torch.Tensor:
        # shape: [batch_size, max_seq_len_per_batch, hidden_size]
        # (in this case, hidden_size is 768 for bert)
        out = self.encoder(**x).last_hidden_state

        if self.variant in ["a", "d"]:
            out = self._fetch_feature_a_or_d(out)
        elif self.variant in ["b", "c", "e"]:
            out = self._fetch_feature_b_or_c_or_e(out, cues)
        elif self.variant == "f":
            out = self._fetch_feature_f(out, cues)

        out = self.fc(out)
        return out

    def _fetch_feature_a_or_d(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Fetch feature for variant 'a' or 'd', i.e. gather the embeddings
        at [CLS] positions.

        Args:
            embeddings: The text embeddings of shape `[batch_size, max_seq_len_per_batch,
            hidden_size]`.

        Returns:
            The [CLS] embedding, of shape `[batch_size, hidden_size]`.
        """
        return torch.squeeze(embeddings[:, 0, :], dim=1)

    def _fetch_feature_b_or_c_or_e(
        self, embeddings: torch.Tensor, cues: Tuple[torch.Tensor]
    ) -> torch.Tensor:
        """Fetch feature for variant 'a' or 'd', i.e. gather the embeddings at
        entity spans, perform max-pooling respectively, and concatenate them.

        Args:
            embeddings: The text embeddings of shape `[batch_size, max_seq_len_per_batch,
            hidden_size]`.
            cues: The start and end positions of both e1 and e2, of shape `[4, batch_size]`.

        Returns:
            The concatenation of the two max-pooled embeddings, of shape `[batch_size,
            2 * hidden_size].
        """
        start_e1, end_e1, start_e2, end_e2 = cues
        embedding_e1 = self._get_pooled_embedding(embeddings, start_e1, end_e1)
        embedding_e2 = self._get_pooled_embedding(embeddings, start_e2, end_e2)
        return torch.cat((embedding_e1, embedding_e2), dim=1)

    def _fetch_feature_f(
        self, embeddings: torch.Tensor, cues: Tuple[torch.Tensor]
    ) -> torch.Tensor:
        """Fetch feature for variant 'f', i.e. gather the embeddings at entity-start cues.

        Args:
            embeddings: Of shape `[batch_size, max_seq_len_per_batch, hidden_size]`.
            cues: the starts of e1 and e2, of shape `[2, batch_size]`.

        Returns:
            The concatenation of the two start embeddings, of shape `[batch_size, 2 * hidden_size].
        """
        # both two vectors have shape `[batch_size, ]`
        start_e1, start_e2 = cues

        # get the embedding of two entity starts, each of shape `[batch_size, hidden_size]`
        embedding_e1 = embeddings[torch.arange(len(embeddings)), start_e1]
        embedding_e2 = embeddings[torch.arange(len(embeddings)), start_e2]
        return torch.cat((embedding_e1, embedding_e2), dim=1)

    @staticmethod
    def _get_pooled_embedding(
        embeddings: torch.Tensor,
        starts: torch.Tensor,
        ends: torch.Tensor,
    ) -> torch.Tensor:
        """Given the embedding of the whole text, fetch the spans of one entity and apply max-
        pooling method to get a fix-sized representation w.r.t. that entity.

        Args:
            embeddings: The text embeddings of shape `[batch_size, max_seq_len_per_batch,
            hidden_size]`.
            starts: The start of the entity, of shape `[batch_size, ]`.
            ends: The end of the entity, of shape `[batch_size, ]`.

        Returns:
            The concatenation of the two max-pooled embeddings, of shape `[batch_size,
            2 * hidden_size].
        """
        features: List[torch.Tensor] = []
        for embedding, start, end in zip(embeddings, starts, ends):
            pooled_embedding = torch.max(
                embedding[
                    start : (end + 1),
                ],
                dim=0,
                keepdim=True,
            ).values
            features.append(pooled_embedding)
        return torch.cat(features)
