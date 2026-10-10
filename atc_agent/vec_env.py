"""
Stable-Baselines3 ``VecEnv`` over several worker processes, each running one
``BatchedCompetitionSim`` (BlueSky is a per-process singleton, so parallelism
across CPU cores needs processes; parallelism inside a process comes from
batching scenarios into one simulator).
"""

import contextlib
import io
import multiprocessing as mp
import os

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from atc_agent.actions import action_space
from atc_agent.features import OBS_DIM


def _worker(remote, parent_remote, cfg):
    parent_remote.close()
    with contextlib.redirect_stdout(io.StringIO()):     # BlueSky is chatty at init
        from atc_agent.batched_env import BatchedCompetitionSim
        sim = BatchedCompetitionSim(**cfg)
    remote.send(("ready", sim.n_seats, sim.seat_is_ma.copy()))
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                remote.send(sim.step(data))
            elif cmd == "reset":
                remote.send(sim.reset())
            elif cmd == "set_reward":
                sim.set_reward_params(data)
                remote.send(True)
            elif cmd == "get_reward":
                remote.send(dict(sim.reward))
            elif cmd == "close":
                break
            else:
                raise ValueError(cmd)
    except (KeyboardInterrupt, EOFError, BrokenPipeError):
        pass
    finally:
        remote.close()


class BlueSkyBatchVecEnv(VecEnv):
    """One VecEnv entry per controlled aircraft ("seat") across all workers."""

    def __init__(self, worker_cfgs, start_method=None):
        start_method = start_method or ("spawn" if os.name == "nt" else "forkserver")
        ctx = mp.get_context(start_method)
        self.remotes, self.procs = [], []
        saved = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
        for k in saved:
            os.environ[k] = "1"      # numpy in workers: one thread each
        try:
            for i, cfg in enumerate(worker_cfgs):
                remote, work_remote = ctx.Pipe()
                p = ctx.Process(target=_worker, args=(work_remote, remote, cfg), daemon=True)
                p.start()
                work_remote.close()
                self.remotes.append(remote)
                self.procs.append(p)
                if i == 0:           # first worker alone: lets BlueSky build its cache files once
                    first = remote.recv()
            msgs = [first] + [r.recv() for r in self.remotes[1:]]
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.seats = [m[1] for m in msgs]
        self.seat_is_ma = np.concatenate([m[2] for m in msgs])
        self.offsets = np.concatenate([[0], np.cumsum(self.seats)])
        obs_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        super().__init__(int(self.offsets[-1]), obs_space, action_space())
        self._actions = None

    # ---------------------------------------------------------- VecEnv API
    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        return np.concatenate([r.recv() for r in self.remotes])

    def step_async(self, actions):
        actions = np.asarray(actions)
        for w, r in enumerate(self.remotes):
            r.send(("step", actions[self.offsets[w]:self.offsets[w + 1]]))

    def step_wait(self):
        results = [r.recv() for r in self.remotes]
        obs = np.concatenate([x[0] for x in results])
        rew = np.concatenate([x[1] for x in results])
        dones = np.concatenate([x[2] for x in results])
        infos = [{} for _ in range(self.num_envs)]
        for w, x in enumerate(results):
            off = int(self.offsets[w])
            for seat, info in x[4].items():
                infos[off + seat] = info
        return obs, rew, dones, infos

    def set_reward_params(self, params):
        for r in self.remotes:
            r.send(("set_reward", dict(params)))
        for r in self.remotes:
            r.recv()

    def get_reward_params(self):
        self.remotes[0].send(("get_reward", None))
        return self.remotes[0].recv()

    def close(self):
        if self.closed:
            return
        for r in self.remotes:
            try:
                r.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        self.closed = True

    def get_attr(self, attr_name, indices=None):
        if attr_name == "seat_is_ma":
            vals = self.seat_is_ma
        elif attr_name == "render_mode":
            vals = [None] * self.num_envs
        else:
            raise AttributeError(attr_name)
        idx = self._get_indices(indices)
        return [vals[i] for i in idx]

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError("use set_reward_params")

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._get_indices(indices))

    def get_images(self):
        return [None] * self.num_envs

    closed = False
