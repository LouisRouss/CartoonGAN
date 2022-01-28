import os
from .trainer import Trainer

def test(config):
    trainer = Trainer(config)
    trainer.test()
    print('test complete')