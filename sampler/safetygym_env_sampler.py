import numpy as np

class SafetygymEnvSampler():
    def __init__(self, args, env, max_path_length=400):
        self.env = env
        self.args = args
        self.path_length = 0
        self.total_path_length = 0
        self.current_state = None
        self.max_path_length = max_path_length
        self.needs_reset = True
        self.last_terminated = False  # NEW

    def sample(self, agent, i, eval_t=False):
        self.total_path_length += 1

        if self.needs_reset or self.current_state is None:
            obs, info = self.env.reset()
            self.current_state = obs
            self.path_length = 0

        cur_state = self.current_state
        action = agent.select_action(cur_state, eval_t)

        next_state, reward, binary_cost, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        self.last_terminated = bool(terminated)  # NEW

        if not eval_t:
            done = False if i == self.args.epoch_length - 1 or "TimeLimit.truncated" in info else done
            done = True if "goal_met" in info and info["goal_met"] else done

        self.path_length += 1
        # CAL's constrained objective expects a non-negative violation cost.
        # The continuous signed distance is still available as info["h"].
        reward = np.array([reward, binary_cost], dtype=np.float32)

        if terminated or truncated:
            self.needs_reset = True
            self.current_state = None
        else:
            self.needs_reset = False
            self.current_state = next_state

        return cur_state, action, next_state, reward, done, info
