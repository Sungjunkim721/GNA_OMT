import torch

from gna_omt.losses import constraint_loss, material_redundancy_loss


def test_redundant_pair_has_positive_penalty():
    m = torch.tensor([[2.1, 2.2]], requires_grad=True)
    loss = material_redundancy_loss(m)
    assert loss.item() > 0
    loss.backward()
    assert m.grad[0, 0].item() == 0.0  # rounded reference is detached
    assert m.grad[0, 1].abs().item() > 0.0


def test_constraint_mask_excludes_unselected_layers():
    t = torch.tensor([[0.0, 0.5, 0.8]], requires_grad=True)
    m = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    target_t = torch.tensor([1.0, 100.0, 100.0])
    target_m = torch.tensor([4.0, 100.0, 100.0])
    mask = torch.tensor([1.0, 0.0, 0.0])
    loss = constraint_loss(t, m, target_t, target_m, mask, mask)
    loss.backward()
    assert t.grad[0, 0].abs().item() > 0
    assert torch.all(t.grad[0, 1:] == 0)
    assert m.grad[0, 0].abs().item() > 0
    assert torch.all(m.grad[0, 1:] == 0)
