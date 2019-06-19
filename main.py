from runner import *
from GUI.net_arch_window import ChoiceWindow

# run_mnist_feedforward_network(
#         hidden_shape=[512],
#         learning_rate=0.1,
#         is_agent_mode_enabled=True
# )

# run_cnn(
#         learning_rate=0.01,
#         batch_size=50
# )

def set_architecture(architecture):
    run_xor_feedforward_network(architecture, False)

b = ChoiceWindow(set_architecture)
b.show_architecture_choice()