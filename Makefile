smoke_test:
	uv run train mytask --env.scene.num-envs 64 --agent.max_iterations 5

train_test:
	uv run train mytask --env.scene.num-envs 2048 --agent.max_iterations $(iter)

look_result:
	uv run tensorboard --logdir logs/rsl_rl/velocity

train_test_continue:
	uv run train mytask --env.scene.num-envs 2048 \
	--agent.resume True \
	--agent.load_run $(start_velocity) \
	--agent.load_checkpoint $(modelpt) \
	--agent.max_iterations $(max_iterations)

play_models:
	uv run play mytask \
		--checkpoint_file logs/rsl_rl/velocity/$(dir)/$(modelpt)
