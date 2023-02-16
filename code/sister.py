import numpy as np
from neural_network import NeuralNetwork

class Sister(NeuralNetwork):
    def __init__(self, W1: np.ndarray, W2: np.ndarray, B1: np.ndarray, B2: np.ndarray,
            dims: dict, bloodline: int, num_food: int):

        super().__init__(W1,W2,B1,B2,dims)

        self.bloodline = bloodline #  what sisters it belongs to
        self.num_food = num_food #  how much food the indvidual has
        self.sis = 0 #  how many acts of altruism to kin
        self.non_sis = 0 #  how much act of altruism to non kin

        self.sis_encounter = 0
        self.non_sis_encounter = 0

        # amount of times acted egotistically
        self.non_sis_ego = 0
        self.sis_ego = 0

    # collect output and round to nearest binary value (0 or 1)
    def output(self, input: np.ndarray):

        output = super().output(input)

        decision = 0 if output < 0.5 else 1

        # decision = np.where(output < 0.5, output, 1)
        return decision

    # update acts of altruism to sisters
    def alt_sis(self,value):
        self.sis += value

    # update acts of altruism to non sisters
    def alt_non_sis(self,value):
        self.non_sis += value

    # can be used to decrement or increment amount of food the indvidual has
    def add_num_food(self,value):
        self.num_food += value
