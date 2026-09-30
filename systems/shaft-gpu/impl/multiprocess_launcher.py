#!/usr/bin/env python3
"""MultiProcessLauncher for CrypTen, copied from SHAFT examples with minor fixes."""

import logging
import multiprocessing
import os
import uuid
import crypten
import torch


class MultiProcessLauncher:
    def __init__(self, world_size, run_process_fn, fn_args=None):
        env = os.environ.copy()
        env["WORLD_SIZE"] = str(world_size)
        multiprocessing.set_start_method("spawn", force=True)

        INIT_METHOD = "file:///tmp/crypten-rendezvous-{}".format(uuid.uuid1())
        env["RENDEZVOUS"] = INIT_METHOD

        self.processes = []
        for rank in range(world_size):
            process = multiprocessing.Process(
                target=self.__class__._run_process,
                name="process " + str(rank),
                args=(rank, world_size, env, run_process_fn, fn_args),
            )
            self.processes.append(process)

        if crypten.mpc.ttp_required():
            ttp_process = multiprocessing.Process(
                target=self.__class__._run_process,
                name="TTP",
                args=(world_size, world_size, env,
                      crypten.mpc.provider.TTPServer, None),
            )
            self.processes.append(ttp_process)

    @classmethod
    def _run_process(cls, rank, world_size, env, run_process_fn, fn_args):
        for env_key, env_value in env.items():
            os.environ[env_key] = env_value
        os.environ["RANK"] = str(rank)
        orig_logging_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.INFO)
        crypten.init()
        # DETERMINISTIC SEEDING: crypten.init() seeds from os.urandom(8), so without this the shares
        # and reconstructed logits differ every run and the gate is unusable; the seed is part of
        # the measured system (MANIFEST). manual_seed() refuses outside debug mode, so debug is
        # toggled on only for the seeding call.
        from crypten.config import cfg as _cfg
        _dbg = _cfg.debug.debug_mode
        _cfg.debug.debug_mode = True
        crypten.manual_seed(0xDEADBEEF + rank, 0xC0FFEE + rank, 0x5EED)
        _cfg.debug.debug_mode = _dbg
        torch.manual_seed(0x5EED)
        logging.getLogger().setLevel(orig_logging_level)
        if fn_args is None:
            run_process_fn()
        else:
            run_process_fn(fn_args)

    def start(self):
        for process in self.processes:
            process.start()

    def join(self):
        for process in self.processes:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(
                    f"Process {process.name} exited with code {process.exitcode}"
                )

    def terminate(self):
        for process in self.processes:
            process.terminate()
