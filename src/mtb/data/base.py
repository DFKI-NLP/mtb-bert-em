from typing import Union, List, Dict, Any

import torch
import datasets


class REDataset(torch.utils.data.Dataset):
    """A general relation extraction (RE) dataset from the raw dataset."""

    def __init__(
        self,
        data_file: str,
        entity_marker: bool = True,
        text_column_name: str = "token",
        label_column_name: str = "relation",
    ):
        """
        Args:
            data_file: Path to the .json file for the split of data.
            entity_marker: A boolean to indicate whether or not to insert entity markers ("<e1>",
            "</e1>", "<e2>", "</e2>") to the original text. The `True` case corresponds to variants
            "a", "b" and "c".
            text_colomn_name: The name of the column for the text, e.g. "token" for TACRED.
            label_column_name: The name of the column for the label, e.g. "relation" for TACRED.
        """
        super().__init__()
        self.dataset = datasets.load_dataset(
            "json", data_files=data_file, split="train"
        )
        self.entity_marker = entity_marker
        self.text_column_name = text_column_name
        self.label_column_name = label_column_name

        self.label_to_id = self._get_label_to_id()
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        self.add_column_for_label_id(new_column_name="relation_id")

    def __getitem__(self, index: Union[int, List[int], torch.Tensor]):
        if torch.is_tensor(index):
            index = index.tolist()
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def _get_label_to_id(self) -> Dict[str, int]:
        """Get a dict of the class-id mapping."""
        label_list = list(set(self.dataset[self.label_column_name]))
        label_list.sort()
        return {label: i for i, label in enumerate(label_list)}

    def add_column_for_label_id(self, new_column_name: str = "relation_id") -> None:
        """Add a new column to store the (relation) class ids mapped from the relation names."""
        new_column_features = [
            self.label_to_id[label] for label in self.dataset[self.label_column_name]
        ]
        self.dataset = self.dataset.add_column(new_column_name, new_column_features)

    @staticmethod
    def convert_special_tokens(
        example: Dict[str, Any],
        text_column_name: str = "token",
        special_tokens_dict: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """A datasets.Dataset processor to convert special tokens to natural language tokens."""
        converted_tokens = []
        for token in example[text_column_name]:
            if token.lower() in special_tokens_dict:
                converted_tokens.append(special_tokens_dict[token.lower()])
            else:
                converted_tokens.append(token)
        example[text_column_name] = converted_tokens
        return example

    @staticmethod
    def insert_entity_markers(
        example: Dict[str, Any], text_column_name: str = "token"
    ) -> Dict[str, Any]:
        tokens = []
        for idx, token in enumerate(example[text_column_name]):
            if idx == example["subj_start"]:
                tokens.append("<e1>")
            elif idx == example["obj_start"]:
                tokens.append("<e2>")
            tokens.append(token)
            if idx == example["subj_end"]:
                tokens.append("</e1>")
            elif idx == example["obj_end"]:
                tokens.append("</e2>")
        example[text_column_name] = tokens

        # update start and end whose span includes entity markers
        # Note: we presume there is no overlap between the two entity spans
        if example["subj_start"] < example["obj_start"]:
            example["subj_end"] += 2
            example["obj_start"] += 2
            example["obj_end"] += 4
        else:
            example["obj_end"] += 2
            example["subj_start"] += 2
            example["subj_end"] += 4

        return example
