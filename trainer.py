import model


class TetrisTrainer():

    def __init__(self):

        self.q_network = model.TetrisNetwork(input_dim=200, hidden_size=32, output_dim=5)


    def optimize(self):
        pass


    def select_action(self):
        pass

    # ToDo later:
    # - Save/checkpoint our training progess  =  saving the weights of the network to a file
    # - Load the weights
    # - Select best action
    # - Select actions that allow us to explore the state space
    # - Stop function at the end of training
