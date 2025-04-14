from transformers import TrainerCallback


class EpochBasedUnfreezeCallback(TrainerCallback):
    """
    在训练指定 epoch 后自动解冻 original_mlp 参数。
    """
    def __init__(self, unfreeze_original_mlp_epoch: int = 1):
        self.unfreeze_original_mlp_epoch = unfreeze_original_mlp_epoch
        self.has_unfrozed = False
    
    def on_epoch_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        if not self.has_unfrozed and state.epoch >= self.unfreeze_original_mlp_epoch:
            for name, param in model.named_parameters():
                if "original_mlp" in name:
                    param.requires_grad = True
                if param.requires_grad:
                    print(name)

            print(f"[Epoch {state.epoch}] Unfroze original_mlp parameters.")
            self.has_unfrozed = True
        return control
    