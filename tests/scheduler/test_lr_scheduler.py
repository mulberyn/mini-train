import math
import pytest
from trainer.utils.lr_scheduler import LRScheduler


def test_initial_step():
    scheduler = LRScheduler(max_lr=1e-3, min_lr=1e-5, warmup_steps=10, total_steps=100)
    assert scheduler.current_step == 0


def test_initial_lr_is_zero_during_warmup():
    scheduler = LRScheduler(max_lr=1e-3, min_lr=1e-5, warmup_steps=10, total_steps=100)
    assert scheduler.get_lr() == pytest.approx(0.0)


def test_step_increments_current_step():
    scheduler = LRScheduler(max_lr=1e-3, min_lr=1e-5, warmup_steps=10, total_steps=100)
    scheduler.step()
    assert scheduler.current_step == 1


def test_linear_warmup():
    max_lr = 1e-3
    warmup_steps = 10
    scheduler = LRScheduler(max_lr=max_lr, min_lr=1e-5, warmup_steps=warmup_steps, total_steps=100)
    for step in range(warmup_steps):
        expected = max_lr * step / warmup_steps
        assert scheduler.get_lr() == pytest.approx(expected)
        scheduler.step()


def test_lr_reaches_max_lr_after_warmup():
    scheduler = LRScheduler(max_lr=1e-3, min_lr=1e-5, warmup_steps=10, total_steps=100)
    for _ in range(10):
        scheduler.step()
    assert scheduler.get_lr() == pytest.approx(1e-3)


def test_cosine_decay():
    max_lr = 1e-3
    min_lr = 1e-5
    warmup_steps = 10
    total_steps = 110
    scheduler = LRScheduler(max_lr=max_lr, min_lr=min_lr, warmup_steps=warmup_steps, total_steps=total_steps)
    for _ in range(warmup_steps):
        scheduler.step()
    assert scheduler.get_lr() == pytest.approx(max_lr)
    for _ in range(50):
        scheduler.step()
    expected = min_lr + 0.5 * (max_lr - min_lr)
    assert scheduler.get_lr() == pytest.approx(expected)


def test_lr_reaches_min_lr():
    scheduler = LRScheduler(max_lr=1e-3, min_lr=1e-5, warmup_steps=10, total_steps=100)
    for _ in range(100):
        scheduler.step()
    assert scheduler.get_lr() == pytest.approx(1e-5)


def test_lr_stays_at_min_lr_after_total_steps():
    scheduler = LRScheduler(max_lr=1e-3, min_lr=1e-5, warmup_steps=10, total_steps=100)
    for _ in range(150):
        scheduler.step()
    assert scheduler.get_lr() == pytest.approx(1e-5)


def test_warmup_is_monotonically_increasing():
    scheduler = LRScheduler(max_lr=1e-3, min_lr=1e-5, warmup_steps=20, total_steps=100)
    lrs = []
    for _ in range(20):
        lrs.append(scheduler.get_lr())
        scheduler.step()
    assert all(lrs[i] <= lrs[i + 1] for i in range(len(lrs) - 1))


def test_cosine_decay_is_monotonically_decreasing():
    scheduler = LRScheduler(max_lr=1e-3, min_lr=1e-5, warmup_steps=10, total_steps=100)
    for _ in range(10):
        scheduler.step()
    lrs = []
    for _ in range(90):
        lrs.append(scheduler.get_lr())
        scheduler.step()
    assert all(lrs[i] >= lrs[i + 1] for i in range(len(lrs) - 1))


def test_no_warmup():
    scheduler = LRScheduler(max_lr=1e-3, min_lr=1e-5, warmup_steps=0, total_steps=100)
    assert scheduler.get_lr() == pytest.approx(1e-3)


def test_no_warmup_no_total_steps():
    scheduler = LRScheduler(max_lr=1e-3, min_lr=1e-5, warmup_steps=0, total_steps=None)
    assert scheduler.get_lr() == pytest.approx(1e-3)
    scheduler.step()
    assert scheduler.get_lr() == pytest.approx(1e-3)


def test_no_total_steps():
    scheduler = LRScheduler(max_lr=1e-3, min_lr=1e-5, warmup_steps=10, total_steps=None)
    for _ in range(10):
        scheduler.step()
    assert scheduler.get_lr() == pytest.approx(1e-3)
    for _ in range(100):
        scheduler.step()
    assert scheduler.get_lr() == pytest.approx(1e-3)


def test_state_dict():
    scheduler = LRScheduler(max_lr=1e-3, min_lr=1e-5, warmup_steps=10, total_steps=100)
    for _ in range(37):
        scheduler.step()
    state = scheduler.state_dict()
    assert state["current_step"] == 37
    assert state["max_lr"] == 1e-3
    assert state["min_lr"] == 1e-5
    assert state["warmup_steps"] == 10
    assert state["total_steps"] == 100


def test_load_state_dict():
    scheduler1 = LRScheduler(max_lr=1e-3, min_lr=1e-5, warmup_steps=10, total_steps=100)
    for _ in range(37):
        scheduler1.step()
    state = scheduler1.state_dict()
    scheduler2 = LRScheduler(max_lr=2e-3, min_lr=2e-5, warmup_steps=5, total_steps=200)
    scheduler2.load_state_dict(state)
    assert scheduler2.current_step == scheduler1.current_step
    assert scheduler2.max_lr == scheduler1.max_lr
    assert scheduler2.min_lr == scheduler1.min_lr
    assert scheduler2.warmup_steps == scheduler1.warmup_steps
    assert scheduler2.total_steps == scheduler1.total_steps
    assert scheduler2.get_lr() == pytest.approx(scheduler1.get_lr())


def test_resume_produces_same_lr():
    scheduler1 = LRScheduler(max_lr=1e-3, min_lr=1e-5, warmup_steps=10, total_steps=100)
    for _ in range(50):
        scheduler1.step()
    state = scheduler1.state_dict()
    scheduler2 = LRScheduler(max_lr=1e-3, min_lr=1e-5, warmup_steps=10, total_steps=100)
    scheduler2.load_state_dict(state)
    assert scheduler1.get_lr() == pytest.approx(scheduler2.get_lr())


def test_invalid_learning_rate():
    with pytest.raises(ValueError):
        LRScheduler(max_lr=0, min_lr=1e-5, warmup_steps=10, total_steps=100)


def test_max_lr_must_be_greater_than_min_lr():
    with pytest.raises(ValueError):
        LRScheduler(max_lr=1e-5, min_lr=1e-3, warmup_steps=10, total_steps=100)


def test_negative_min_lr():
    with pytest.raises(ValueError):
        LRScheduler(max_lr=1e-3, min_lr=-1e-5, warmup_steps=10, total_steps=100)


def test_negative_warmup_steps():
    with pytest.raises(ValueError):
        LRScheduler(max_lr=1e-3, min_lr=1e-5, warmup_steps=-1, total_steps=100)


def test_invalid_total_steps():
    with pytest.raises(ValueError):
        LRScheduler(max_lr=1e-3, min_lr=1e-5, warmup_steps=100, total_steps=100)


def test_negative_current_step():
    with pytest.raises(ValueError):
        LRScheduler(max_lr=1e-3, min_lr=1e-5, warmup_steps=10, total_steps=100, current_step=-1)