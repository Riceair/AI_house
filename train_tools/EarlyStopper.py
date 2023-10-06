import numpy as np

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, best_criterion='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.best_criterion = best_criterion
        self.counter = 0
        if best_criterion == 'min':
            self.__record_criterion = np.inf
        elif best_criterion == 'max':
            self.__record_criterion = -np.inf
        else:
            raise NameError("best_criterion must be \'max\' or \'min\'")

    def stopJudgment(self, criterion_value):
        if self.best_criterion == 'min':
            if criterion_value < self.__record_criterion:
                self.__record_criterion = criterion_value
                self.counter = 0
            elif criterion_value >= (self.__record_criterion + self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    return True
            return False
        elif self.best_criterion == 'max':
            if criterion_value > self.__record_criterion:
                self.__record_criterion = criterion_value
                self.counter = 0
            elif criterion_value <= (self.__record_criterion + self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    return True
            return False
# early_stopper = EarlyStopper(patience=3, min_delta=10)
# for epoch in np.arange(n_epochs):
#     train_loss = train_one_epoch(model, train_loader)
#     criterion_value = validate_one_epoch(model, validation_loader)
#     if early_stopper.early_stop(criterion_value):             
#         break