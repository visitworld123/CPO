"""Small smoke test for the adaptive CPO objective."""

import torch


def test_easy_cpo_loss_is_finite():
    beta, gamma = 2.5, 0.05
    chosen = torch.tensor([-1.0, -2.0])
    rejected = torch.tensor([-1.5, -1.0])
    ref_chosen = torch.tensor([-1.2, -1.8])
    ref_rejected = torch.tensor([-1.4, -1.1])
    policy_logratio = chosen - rejected
    ref_logratio = ref_chosen - ref_rejected
    margin = gamma * (1 / ref_chosen + 1 / ref_rejected)
    loss = -torch.nn.functional.logsigmoid(beta * policy_logratio - beta * ref_logratio - margin)
    assert torch.isfinite(loss).all()
    assert loss.shape == chosen.shape


if __name__ == "__main__":
    test_easy_cpo_loss_is_finite()
    print("CPO loss smoke test passed")
