from transformers import TrainerCallback


class EpochBasedUnfreezeCallback(TrainerCallback):
    """
    在训练指定 epoch 后自动解冻 shared 参数。
    """
    def __init__(self, unfreeze_shared_epoch: int = 1):
        self.unfreeze_shared_epoch = unfreeze_shared_epoch
        self.has_unfrozed = False
    
    def on_epoch_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        if not self.has_unfrozed and state.epoch >= self.unfreeze_shared_epoch:
            for name, param in model.named_parameters():
                if "shared" in name:
                    param.requires_grad = True
                if param.requires_grad:
                    print(name)

            print(f"[Epoch {state.epoch}] Unfroze shared parameters.")
            self.has_unfrozed = True
        return control
    