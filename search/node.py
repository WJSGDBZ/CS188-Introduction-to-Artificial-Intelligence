class Node:

    def __init__(self, st, ac=None, bwc=0):
        if ac is None:
            ac = []
        else:
            self.actions = ac  # collection of action
        self.state = st  # location tuple(x, y)
        self.cost = bwc  # total backward cost

    def get_state(self):
        return self.state

    def add_action(self, action):
        self.actions.append(action)

    def get_actions(self):
        return self.actions

    def get_cost(self):
        return self.cost

    def add_cost(self, cost):
        self.cost += cost
