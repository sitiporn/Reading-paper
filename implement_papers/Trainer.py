
class Trainer:
    """
    Trainer is a simple but feature-complete training and eval loop for PyTorch, optimized for 🤗 Transformers.
    Args:
        model (:class:`~transformers.PreTrainedModel` or :obj:`torch.nn.Module`, `optional`):
            The model to train, evaluate or use for predictions. If not provided, a ``model_init`` must be passed.
    """

     # Label smoothing
     if self.args.label_smoothing_factor != 0:
     self.label_smoother = LabelSmoother(epsilon=self.args.label_smoothing_factor)
     else:
     self.label_smoother = None

     self.state = TrainerState()
     self.control = TrainerControl()
     # Internal variable to count flos in each process, will be accumulated in `self.state.total_flos` then
     # returned to 0 every time flos need to be logged
     self.current_flos = 0
     self.hp_search_backend = None
     self.use_tune_checkpoints = False
     default_label_names = (
     ["start_positions", "end_positions"]
     if type(self.model).__name__ in MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES.values()
     else ["labels"]
     )
     self.label_names = default_label_names if self.args.label_names is None else self.args.label_names
     self.control = self.callback_handler.on_init_end(self.args, self.state, self.control)


     def get_train_dataloader(self) -> DataLoader:
            """
            Returns the training :class:`~torch.utils.data.DataLoader`.
            Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
            to distributed training if necessary) otherwise.
            Subclass and override this method if you want to inject some custom behavior.
            """
            if self.train_dataset is None:
                raise ValueError("Trainer: training requires a train_dataset.")

            train_dataset = self.train_dataset
            if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
                train_dataset = self._remove_unused_columns(train_dataset, description="training")

            if isinstance(train_dataset, torch.utils.data.IterableDataset):
                if self.args.world_size > 1:
                    train_dataset = IterableDatasetShard(
                        train_dataset,
                        batch_size=self.args.train_batch_size,
                        drop_last=self.args.dataloader_drop_last,
                        num_processes=self.args.world_size,
                        process_index=self.args.process_index,
                    )

                return DataLoader(
                    train_dataset,
                    batch_size=self.args.train_batch_size,
                    collate_fn=self.data_collator,
                    num_workers=self.args.dataloader_num_workers,
                    pin_memory=self.args.dataloader_pin_memory,
                )

            train_sampler = self._get_train_sampler()

            return DataLoader(
                train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=train_sampler,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.
        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
