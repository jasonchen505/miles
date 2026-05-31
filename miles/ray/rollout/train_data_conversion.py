import ray
import torch

from miles.utils.ray_utils import Box
from miles.utils.seqlen_balancing import get_seqlen_balanced_partitions
from miles.utils.types import Sample


# TODO: remove `self`
def convert_samples_to_train_data(self, samples: list[Sample] | list[list[Sample]]):
    """
    Convert inference generated samples to training data.
    """
    if self.custom_convert_samples_to_train_data_func is not None:
        return self.custom_convert_samples_to_train_data_func(self.args, samples)

    raw_rewards, rewards = _post_process_rewards(self, samples)

    assert len(raw_rewards) == len(samples)
    assert len(rewards) == len(samples)

    train_data = {
        "tokens": [sample.tokens for sample in samples],
        "response_lengths": [sample.response_length for sample in samples],
        # some reward model, e.g. remote rm, may return multiple rewards,
        # we could use key to select the reward.
        "rewards": rewards,
        "raw_reward": raw_rewards,
        "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
        "sample_indices": [sample.index for sample in samples],
    }

    # loss mask
    # TODO: compress the loss mask
    loss_masks = []
    for sample in samples:
        # always instantiate loss_mask if not provided
        if sample.loss_mask is None:
            sample.loss_mask = [1] * sample.response_length

        assert (
            len(sample.loss_mask) == sample.response_length
        ), f"loss mask length {len(sample.loss_mask)} != response length {sample.response_length}"
        if sample.remove_sample:
            sample.loss_mask = [0] * sample.response_length
        loss_masks.append(sample.loss_mask)
    train_data["loss_masks"] = loss_masks

    # overwriting the raw reward
    if samples[0].metadata and "raw_reward" in samples[0].metadata:
        train_data["raw_reward"] = [sample.metadata["raw_reward"] for sample in samples]

    # For rollout buffer
    if samples[0].metadata and "round_number" in samples[0].metadata:
        train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]

    # Add rollout log probabilities for off-policy correction
    if samples[0].rollout_log_probs is not None:
        train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]

    if samples[0].rollout_routed_experts is not None:
        train_data["rollout_routed_experts"] = [sample.rollout_routed_experts for sample in samples]

    if samples[0].train_metadata is not None:
        train_data["metadata"] = [sample.train_metadata for sample in samples]

    if any(sample.multimodal_train_inputs is not None for sample in samples):
        train_data["multimodal_train_inputs"] = [sample.multimodal_train_inputs for sample in samples]

    if any(sample.weight_versions for sample in samples):
        train_data["weight_versions"] = [sample.weight_versions for sample in samples]

    if "teacher_log_probs" in samples[0].__dict__:
        train_data["teacher_log_probs"] = [sample.teacher_log_probs for sample in samples]

    # Pass dynamic global_batch_size to training side
    assert self.args.use_dynamic_global_batch_size == hasattr(self, "_dynamic_global_batch_size")
    if hasattr(self, "_dynamic_global_batch_size"):
        train_data["dynamic_global_batch_size"] = self._dynamic_global_batch_size

    return train_data


# TODO: remove `self`
def _post_process_rewards(self, samples: list[Sample] | list[list[Sample]]):
    if self.custom_reward_post_process_func is not None:
        return self.custom_reward_post_process_func(self.args, samples)

    raw_rewards = [sample.get_reward_value(self.args) for sample in samples]
    if (
        self.args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
        and self.args.rewards_normalization
    ):
        # group norm
        rewards = torch.tensor(raw_rewards, dtype=torch.float)
        if rewards.shape[-1] == self.args.n_samples_per_prompt * self.args.rollout_batch_size:
            rewards = rewards.reshape(-1, self.args.n_samples_per_prompt)
        else:
            # when samples count are not equal in each group
            rewards = rewards.view(-1, rewards.shape[-1])
        mean = rewards.mean(dim=-1, keepdim=True)
        rewards = rewards - mean

        if self.args.advantage_estimator in ["grpo", "gspo"] and self.args.grpo_std_normalization:
            std = rewards.std(dim=-1, keepdim=True)
            rewards = rewards / (std + 1e-6)

        return raw_rewards, rewards.flatten().tolist()

    return raw_rewards, raw_rewards


# TODO: remove `self`
def split_train_data_by_dp(self, data, dp_size):
    """Split the train data by data parallel size."""
    rollout_data = {}

    if "prompt" in data:
        rollout_data["prompt"] = data["prompt"]

    total_lengths = [len(t) for t in data["tokens"]]
    data["total_lengths"] = total_lengths

    if self.args.balance_data:
        partitions = get_seqlen_balanced_partitions(total_lengths, dp_size, equal_size=True)
    else:
        partitions = [range(i, len(total_lengths), dp_size) for i in range(dp_size)]

    rollout_data_refs = []

    for i in range(dp_size):
        rollout_data = {}
        partition = partitions[i]
        rollout_data["partition"] = partition
        for key in [
            "tokens",
            "multimodal_train_inputs",
            "response_lengths",
            "rewards",
            "truncated",
            "loss_masks",
            "round_number",
            "sample_indices",
            "rollout_log_probs",
            "rollout_routed_experts",
            "prompt",
            "teacher_log_probs",
            "weight_versions",
        ]:
            if key not in data:
                continue
            val = [data[key][j] for j in partition]
            rollout_data[key] = val
        # keys that need to be splited at train side
        for key in [
            "raw_reward",
            "total_lengths",
            "dynamic_global_batch_size",
        ]:
            if key not in data:
                continue
            rollout_data[key] = data[key]
        rollout_data_refs.append(Box(ray.put(rollout_data)))
    return rollout_data_refs
