from lightning.pytorch.callbacks import Callback
import torch.optim.lr_scheduler as lr_scheduler

class CustomLRScheduler(Callback):    
    def __init__(self, initial_lr, linear_phase_epochs, stable_phase_epochs, cyclic_phase_epochs):
        super().__init__()
        self.initial_lr = initial_lr
        self.linear_phase_epochs = linear_phase_epochs
        self.stable_phase_epochs = stable_phase_epochs
        self.cyclic_phase_epochs = cyclic_phase_epochs

        # Setting stable learning rate as a percentage drop from initial
        self.stable_lr = initial_lr * 0.75

        # Derive cyclic learning rates based on stable learning rate
        self.cyclic_lr_base = self.stable_lr * 0.8
        self.cyclic_lr_max = self.stable_lr * 1.2

        # Constants for reduce on plateau
        self.reduce_lr_factor = 0.5
        self.reduce_lr_patience = 10

        # Initial phase
        self.current_phase = 'linear_decrease'
        self.scheduler = None
        
    # def __init__(self, initial_lr, linear_decrease_lr, linear_phase_epochs, stable_lr, stable_phase_epochs, cyclic_lr_max, cyclic_lr_base, reduce_lr_factor, reduce_lr_patience, cyclic_phase_epochs):
    #     super().__init__()
    #     self.initial_lr = initial_lr
    #     self.linear_decrease_lr = linear_decrease_lr
    #     self.linear_phase_epochs = linear_phase_epochs
    #     self.stable_lr = stable_lr
    #     self.stable_phase_epochs = stable_phase_epochs
    #     self.cyclic_lr_max = cyclic_lr_max
    #     self.cyclic_lr_base = cyclic_lr_base
    #     self.reduce_lr_factor = reduce_lr_factor
    #     self.reduce_lr_patience = reduce_lr_patience
    #     self.cyclic_phase_epochs = cyclic_phase_epochs
    #     self.current_phase = 'linear_decrease'
    #     self.scheduler = None

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        optimizer = trainer.optimizers[0]

        if self.current_phase == 'linear_decrease':
            lr_decrease = (self.initial_lr - self.stable_lr) * (epoch / self.linear_phase_epochs)
            lr = self.initial_lr - lr_decrease
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            if epoch >= self.linear_phase_epochs - 1:
                self.current_phase = 'stable'

        elif self.current_phase == 'stable':
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.stable_lr
            if epoch >= self.linear_phase_epochs + self.stable_phase_epochs - 1:
                self.current_phase = 'cyclic'
                self.scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=self.cyclic_lr_base, max_lr=self.cyclic_lr_max, step_size_up=4, cycle_momentum=False)

        elif self.current_phase == 'cyclic':
            assert isinstance(self.scheduler, lr_scheduler.CyclicLR), "Scheduler is not an instance of CyclicLR"
            if self.scheduler and epoch < self.linear_phase_epochs + self.stable_phase_epochs + self.cyclic_phase_epochs:
                self.scheduler.step()
            elif epoch >= self.linear_phase_epochs + self.stable_phase_epochs + self.cyclic_phase_epochs:
                self.current_phase = 'plateau'
                self.scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.reduce_lr_factor, patience=self.reduce_lr_patience)

        elif self.current_phase == 'plateau':
            assert isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau), "Scheduler is not an instance of ReduceLROnPlateau"
            if self.scheduler:
                val_loss = trainer.callback_metrics.get('val_loss')
                if val_loss is not None:
                    self.scheduler.step(val_loss)

    def on_train_epoch_end(self, trainer, pl_module, unused=None):
        # Optional: Log the current learning rate
        lr = trainer.optimizers[0].param_groups[0]['lr']
        trainer.logger.log_metrics({'learning_rate': lr}, step=trainer.global_step)
