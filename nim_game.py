import random
import math
import time
import pickle


class Game:

	def __init__(self):
		self.rows = [random.randint(1, 10) for i in range(3)]
		self.player = 0
		self.winner = None
	
	@staticmethod
	def all_actions(rows):
		actions = set()
		for i, row in enumerate(rows):
			for j in range(1, row + 1):
				actions.add((i, j))
		
		return actions
	
	@classmethod
	def other_player(cls, player):
		return 0 if player == 1 else 1
	
	def switch_player(self):
		self.player = Game.other_player(self.player)
	
	def move(self, action):
		row, count = action

		if self.winner is not None:
			raise Exception('Game already won')
		elif row < 0 or row >= len(self.rows):
			raise Exception('Invalid row')
		elif count < 1 or count > self.rows[row]:
			raise Exception('Invalid number of objects')

		self.rows[row] -= count

		if all(row == 0 for row in self.rows):
			self.winner = self.player
		
		self.switch_player()


class NimAI():

	def __init__(self, alpha=0.5, epsilon=0.1):
		self.q = dict()
		self.alpha = alpha
		self.epsilon = epsilon

	def update(self, old_state, action, new_state, reward):
		old = self.get_q_value(old_state, action)
		best_future = self.best_future_reward(new_state)
		self.update_q_value(old_state, action, old, reward, best_future)

	def get_q_value(self, state, action):
		return self.q[tuple(state), action] if (tuple(state), action) in self.q else 0

	def update_q_value(self, state, action, old_q, reward, future_rewards):
		new_q = old_q + self.alpha * (reward + future_rewards - old_q)
		self.q[tuple(state), action] = new_q

	def best_future_reward(self, state):
		actions = Game.all_actions(state)
		best_reward = 0
		for action in actions:
			reward = self.get_q_value(state, action)
			if reward > best_reward:
				best_reward = reward

		return best_reward

	def choose_action(self, state, epsilon=True):
		actions = list(Game.all_actions(state))
		best_action = actions[0]

		for action in actions:
			q_val = self.get_q_value(state, action)

			if q_val > self.get_q_value(state, best_action):
				best_action = action

		if epsilon:
			best_action = random.choices(
				[best_action, random.choice(actions)],
				weights=[1 - self.epsilon, self.epsilon],
				k=1
			)[0]

		return best_action
	
	def save_model(self):
		with open('nim_model.pkl', 'wb') as file:
			data = {
				'q': self.q,
				'alpha': self.alpha,
				'epsilon': self.epsilon
			}
			pickle.dump(data, file)
	
	def load_model(self):
		with open('nim_model.pkl', 'rb') as file:
			data = pickle.load(file)
			self.q = data['q']
			self.alpha = data['alpha']
			self.epsilon = data['epsilon']
	

def train_model(n):
	player = NimAI()
	
	print('Traning begun')
	for i in range(n):
		game = Game()

		last = {
			0: {'state': None, 'action': None},
			1: {'state': None, 'action': None}
		}

		while True:
			state = game.rows.copy()
			action = player.choose_action(game.rows)

			last[game.player]['state'] = state
			last[game.player]['action'] = action
			
			game.move(action)
			new_state = game.rows.copy()

			if game.winner is not None:
				player.update(state, action, new_state, 1)
				player.update(
					last[game.player]['state'],
					last[game.player]['action'],
					new_state,
					-1
				)
				break

			elif last[game.player]['state'] is not None:
				player.update(
					last[game.player]['state'],
					last[game.player]['action'],
					new_state,
					0
				)

	print('Done training')
	
	player.save_model()
	
	return player

def play(ai, human_player=None):
	if human_player is None:
		human_player = random.randint(0, 1)

	game = Game()

	while True:

		print()
		print('rows:')
		for i, row in enumerate(game.rows):
			print(f'row {i}: {"@" * row}')
		print()

		all_actions = Game.all_actions(game.rows)
		time.sleep(1)

		if game.player == human_player:
			print('Your Turn')
			while True:
				row = int(input('Choose row: '))
				count = int(input('Choose Count: '))
				if (row, count) in all_actions:
					break
				print('Invalid move, try again.')

		else:
			print('AI\'s Turn')
			row, count = ai.choose_action(game.rows, epsilon=False)
			print(f'AI chose to take {count} from row {row}.')

		game.move((row, count))

		if game.winner is not None:
			print()
			print('GAME OVER')
			winner = 'Human' if game.winner == human_player else 'AI'
			print(f'Winner is {winner}')
			return
		
ai = NimAI()
ai.load_model()
play(ai)
