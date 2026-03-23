"""Training utilities for running epochs and optimization."""

from typing import Any

from torch import Tensor, argmax, set_grad_enabled
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.utils import RollingCounter


class Trainer:
    """Encapsulate training/evaluation logic for one model."""

    def __init__(
        self,
        model: Any,
        crit: Any,
        opt: Any,
        sch: Any,
        device: str,
    ):
        """Initialize trainer dependencies and runtime device."""
        self.device = device
        self.model = model
        self.crit = crit
        self.opt = opt
        self.sch = sch

    def run_epoch(
        self,
        loader: DataLoader,
        train_mode: bool = True,
    ) -> dict[str, float]:
        """Run a single training or evaluation epoch and return metrics."""
        # Dev Agent Breadcrumb: each batch executes `step`, then metrics are
        # aggregated in a rolling counter for both progress and final summary.
        loss_metric, err_metric = RollingCounter(1000), RollingCounter(1000)
        progress = tqdm(total=len(loader), desc="LR: | Loss: | Err: ")

        self.model.train(mode=train_mode)
        with set_grad_enabled(train_mode):
            for x, y, ignore in loader:
                x = x.to(device=self.device)
                y = y.to(device=self.device)
                ignore = ignore.to(device=self.device)
                loss, err = self.step(x, y, ignore, train_mode)
                loss_metric.add(loss)
                err_metric.add(err)

                progress.set_description(
                    f"LR: {self.sch.get_last_lr()[-1]:.8f} | "
                    f"Loss: {loss_metric.rolling_average():.8f} | "
                    f"Err: {err_metric.rolling_average():.8f}"
                )
                progress.update(1)

        return {
            "total_average_loss": loss_metric.total_average(),
            "rolling_average_loss": loss_metric.rolling_average(),
            "total_average_err": err_metric.total_average(),
            "rolling_average_err": err_metric.rolling_average(),
        }

    def step(
        self,
        x: Tensor,
        y: Tensor,
        ignore: Tensor,
        train_mode: bool = True,
    ) -> tuple[float, float]:
        """Run one forward/backward/optimization step and return loss+error."""
        if train_mode:
            self.model.zero_grad()
            self.opt.zero_grad()

        y_pred, _ = self.model(x, ignore)
        y_pred = y_pred.view(-1, y_pred.size(-1))
        y = y.view(-1)
        loss = self.crit(y_pred, y)

        if train_mode:
            loss.backward()
            clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            self.sch.step()

        y_pred = argmax(y_pred, dim=1)
        err = (y_pred != y).sum() / y.shape[0]

        return loss.item(), err
