from runner import run_feedforward_network, run_cnn

run_feedforward_network(
        hidden_shape=[512],
        learning_rate=0.1,
        is_agent_mode_enabled=False
)

# run_cnn(
#         learning_rate=0.01,
#         batch_size=50
# )