import importlib.util
from pathlib import Path


def module():
    path=Path(__file__).resolve().parents[1]/'scripts/schedule_frozen_baselines.py'
    spec=importlib.util.spec_from_file_location('fixture_scheduler',path)
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_scheduler_uses_spare_slot_without_repeating_finished_or_active():
    scheduler=module()
    methods=['R','H','G'];items=[{'id':str(i)} for i in range(3)]
    done={('R',str(i)) for i in range(3)}
    active={('H','0'),('G','0')}
    assert scheduler.next_job(methods,items,done,active)==('H','1')
    active.add(('H','1'))
    assert scheduler.next_job(methods,items,done,active)==('G','1')


def test_scheduler_fills_all_items_once():
    scheduler=module();methods=['R','H','G'];items=[{'id':str(i)} for i in range(4)]
    finished=set();active=set()
    while True:
        job=scheduler.next_job(methods,items,finished,active)
        if job is None:break
        assert job not in finished
        finished.add(job)
    assert len(finished)==12
