import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
import random
import time
#python=3.8.0, tensorflow=2.8.0, numpy=1.21.0, gym=0.26.2, matplotlib=3.4.3
###############################################Blackjack Ortamı########################################
class BlackjackEnv(gym.Env):
    def __init__(self):
        super(BlackjackEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=31, shape=(2,), dtype=np.float32)
    def reset(self):
        self.player_sum = np.random.randint(12, 22) 
        self.dealer_card = np.random.randint(1, 11)  
        return np.array([self.player_sum, self.dealer_card], dtype=np.float32)
    def step(self, action):
        if action == 0:  
            self.player_sum += np.random.randint(1, 11)
            if self.player_sum==21:
                return np.array([self.player_sum, self.dealer_card], dtype=np.float32), 1, True, {}
            elif self.player_sum > 21:  
                return np.array([self.player_sum, self.dealer_card], dtype=np.float32), -1, True, {}
        else:  
            while self.dealer_card < 17:  
                self.dealer_card += np.random.randint(1, 11)
            if self.dealer_card > 21 or self.player_sum > self.dealer_card:
                return np.array([self.player_sum, self.dealer_card], dtype=np.float32), 1, True, {}
        if(self.dealer_card==self.player_sum):
            return np.array([self.player_sum, self.dealer_card], dtype=np.float32), 0, True, {}
        elif(self.dealer_card>self.player_sum):
            return np.array([self.player_sum, self.dealer_card], dtype=np.float32), -1, True, {}
        elif(self.dealer_card<self.player_sum):
            return np.array([self.player_sum, self.dealer_card], dtype=np.float32), 1, True, {}
################################################SAC Agent##############################################
class SACAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.alpha = 0.2
        self.tau = 0.005
        self.lr = 0.001
        self.batch_size = 64
        self.memory = deque(maxlen=100000)  # Replay buffer

        # Actor ve Critic modelleri
        self.actor_model = self.build_actor()
        self.critic_model_1 = self.build_critic()
        self.critic_model_2 = self.build_critic()
        self.target_critic_model_1 = self.build_critic()
        self.target_critic_model_2 = self.build_critic()
        self.update_target_networks(tau=1.0)

    def build_actor(self):
        model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(self.state_size,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(self.action_size, activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),loss='categorical_crossentropy')  
        return model

    def build_critic(self):
        model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(self.state_size,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),loss='mse')  
        return model
    
    def update_target_networks(self, tau=None):
        """Hedef ağlari güncelle"""
        tau = tau or self.tau
        for target_model, model in zip(
            [self.target_critic_model_1, self.target_critic_model_2],
            [self.critic_model_1, self.critic_model_2],
        ):
            target_weights = target_model.get_weights()
            model_weights = model.get_weights()
            updated_weights = [
                tau * mw + (1 - tau) * tw for mw, tw in zip(model_weights, target_weights)
            ]
            target_model.set_weights(updated_weights)
    def store_experience(self, state, action, reward, next_state, done):
        """Deneyimi replay buffer'a ekle"""
        self.memory.append((state, action, reward, next_state, done))

    def sample_experiences(self):
        """Replay buffer'dan bir batch seç"""
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def train(self):
        """Replay buffer'dan örnekler al ve ağlari güncelle"""
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.sample_experiences()
        # Critic güncelleme
        target_q1 = self.target_critic_model_1.predict(next_states)
        target_q2 = self.target_critic_model_2.predict(next_states)
        target_q = rewards + self.gamma * np.minimum(target_q1, target_q2) * (1 - dones)

        self.critic_model_1.train_on_batch(states, target_q)
        self.critic_model_2.train_on_batch(states, target_q)
        # Actor güncelleme
        with tf.GradientTape() as tape:
            probs = self.actor_model(states)
            q_values = self.critic_model_1(states)
            actor_loss = -tf.reduce_mean(probs * q_values)

        actor_grads = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_model.optimizer.apply_gradients(
            zip(actor_grads, self.actor_model.trainable_variables)
        )
        # Hedef ağları güncelle
        self.update_target_networks()

    def act(self, state):
        state = state.reshape(1, -1)
        prob = self.actor_model.predict(state)[0]
        return np.random.choice(self.action_size, p=prob)


def train_agent_sac(agent, num_train_episodes, env):
    train_win_rates = []
    total_wins = 0

    for episode in range(num_train_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_experience(state, action, reward, next_state, done)
            agent.train()
            total_reward += reward
            state = next_state

        # Kazanılan oyunları say (pozitif ödüller)
        if total_reward > 0:
            total_wins += 1
        win_rate = total_wins / (episode + 1)
        train_win_rates.append(win_rate)

    return train_win_rates


def test_agent_sac(agent, num_test_episodes, env):
    test_win_rates = []
    total_wins = 0

    for episode in range(num_test_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state

        # Kazanılan oyunları say (pozitif ödüller)
        if total_reward > 0:
            total_wins += 1
        win_rate = total_wins / (episode + 1)
        test_win_rates.append(win_rate)

    return test_win_rates


def plot_win_rates_sac(train_win_rates, test_win_rates):
    plt.figure(figsize=(10, 5))
    plt.plot(train_win_rates, label='Train Win Rate', color='blue')
    plt.xlabel('Episodes')
    plt.ylabel('Win Rate')
    plt.title('SAC Agent Train Win Rate')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(test_win_rates, label='Test Win Rate', color='green')
    plt.xlabel('Episodes')
    plt.ylabel('Win Rate')
    plt.title('SAC Agent Test Win Rate')
    plt.legend()
    plt.show()


def CreateSac(num_train, num_test):
    env = BlackjackEnv()
    agent = SACAgent(state_size=2, action_size=env.action_space.n)
    train_win_rates = train_agent_sac(agent, num_train, env)
    test_win_rates = test_agent_sac(agent, num_test, env)
    plot_win_rates_sac(train_win_rates, test_win_rates)

###############################################DDPG Agent#################################################
# Replay Buffer sınıfı
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

# DDPG Ajanı
class DDPGAgent:
    def __init__(self, state_size, action_size, max_action):
        self.state_size = state_size
        self.action_size = action_size
        self.max_action = max_action
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 64

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(max_size=100000)

        # Actor ve Critic modelleri
        self.actor_model = self.build_actor()
        self.critic_model = self.build_critic()
        self.target_actor_model = self.build_actor()
        self.target_critic_model = self.build_critic()

        self.target_actor_model.set_weights(self.actor_model.get_weights())
        self.target_critic_model.set_weights(self.critic_model.get_weights())

    def build_actor(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_size,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='tanh')  # Deterministik aksiyon
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    def build_critic(self):
        state_input = tf.keras.layers.Input(shape=(self.state_size,))
        action_input = tf.keras.layers.Input(shape=(self.action_size,))
        concat = tf.keras.layers.Concatenate()([state_input, action_input])
        dense1 = tf.keras.layers.Dense(128, activation='relu')(concat)
        dense2 = tf.keras.layers.Dense(64, activation='relu')(dense1)
        output = tf.keras.layers.Dense(1)(dense2)
        model = tf.keras.Model(inputs=[state_input, action_input], outputs=output)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        return model

    def act(self, state):
        state = np.expand_dims(state, axis=0)
        action = self.actor_model.predict(state, verbose=0)[0]
        return np.clip(action * self.max_action, -self.max_action, self.max_action)

    def update_target(self):
        for target, main in zip(self.target_actor_model.weights, self.actor_model.weights):
            target.assign(self.tau * main + (1 - self.tau) * target)
        for target, main in zip(self.target_critic_model.weights, self.critic_model.weights):
            target.assign(self.tau * main + (1 - self.tau) * target)

    def train(self):
        if len(self.replay_buffer.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Hedef Q-değerlerini hesapla
        target_actions = self.target_actor_model.predict(next_states, verbose=0)
        target_q_values = self.target_critic_model.predict([next_states, target_actions], verbose=0)
        y = rewards + self.gamma * (1 - dones) * np.squeeze(target_q_values)

        # Critic'i eğit
        self.critic_model.train_on_batch([states, actions], y)

        # Actor'u eğit
        with tf.GradientTape() as tape:
            actions_pred = self.actor_model(states)
            critic_value = self.critic_model([states, actions_pred])
            actor_loss = -tf.reduce_mean(critic_value)
        actor_grads = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_model.optimizer.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables))

        # Hedef ağları güncelle
        self.update_target()

# Eğitim fonksiyonu
def train_agent_ddpq(agent, num_train_episodes):
    env = BlackjackEnv()
    train_win_rates = []
    total_wins = 0

    for episode in range(num_train_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.add((state, action, reward, next_state, done))
            agent.train()
            total_reward += reward
            state = next_state

        if total_reward > 0:
            total_wins += 1
        win_rate = total_wins / (episode + 1)
        train_win_rates.append(win_rate)

    return train_win_rates

# Test fonksiyonu
def test_agent_ddpq(agent, num_test_episodes):
    env = BlackjackEnv()
    test_win_rates = []
    total_wins = 0

    for episode in range(num_test_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state

        if total_reward > 0:
            total_wins += 1
        win_rate = total_wins / (episode + 1)
        test_win_rates.append(win_rate)

    return test_win_rates

# Eğitim ve test sonuçlarını ayrı grafiklerde göster
def plot_win_rates_ddpq(train_win_rates, test_win_rates):
    # Eğitim kazanma oranı grafiği
    plt.figure(figsize=(10, 5))
    plt.plot(train_win_rates, label='Train Win Rate', color='blue')
    plt.xlabel('Episodes')
    plt.ylabel('Win Rate')
    plt.title('DDPG Agent Train Win Rate')
    plt.legend()
    plt.show()

    # Test kazanma oranı grafiği
    plt.figure(figsize=(10, 5))
    plt.plot(test_win_rates, label='Test Win Rate', color='green')
    plt.xlabel('Episodes')
    plt.ylabel('Win Rate')
    plt.title('DDPG Agent Test Win Rate')
    plt.legend()
    plt.show()

# Eğitim ve test süreçlerini başlatan fonksiyon
def Createddpg(num_train, num_test):
    env = BlackjackEnv()
    agent = DDPGAgent(state_size=env.observation_space.shape[0], action_size=1, max_action=1.0)
    train_win_rates = train_agent_ddpq(agent, num_train)
    test_win_rates = test_agent_ddpq(agent, num_test)
    plot_win_rates_ddpq(train_win_rates, test_win_rates)

###################################################PPO Agent##############################################
# PPO Ajanı
class PPOAgent:
    def __init__(self, action_size, clip_ratio=0.2, gamma=0.99, learning_rate=0.001, buffer_size=100000):
        self.action_size = action_size
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        
        self.actor_model = self.build_actor()
        self.critic_model = self.build_critic()
        
        # Buffer tanımlanıyor
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'values': [],
            'next_values': []
        }

    def build_actor(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(2,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def build_critic(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(2,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state):
        state = state.reshape(1, -1)
        prob = self.actor_model.predict(state)[0]
        return np.random.choice(self.action_size, p=prob)

    def compute_advantages(self, rewards, values, next_values, dones):
        advantages = []
        for i in range(len(rewards)):
            delta = rewards[i] + self.gamma * (1 - dones[i]) * next_values[i] - values[i]
            advantages.append(delta)
        return np.array(advantages)

    def update(self, states, actions, advantages, returns):
        actions = np.array(actions)
        advantages = np.array(advantages)
        returns = np.array(returns)

        with tf.GradientTape() as tape:
            probs = self.actor_model(states)
            action_probs = tf.reduce_sum(probs * tf.one_hot(actions, self.action_size), axis=1)
            old_probs = tf.stop_gradient(action_probs)

            ratio = action_probs / (old_probs + 1e-10)
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))

            critic_values = tf.squeeze(self.critic_model(states))
            critic_loss = tf.keras.losses.mean_squared_error(returns, critic_values)
            loss = actor_loss + 0.5 * critic_loss

        grads = tape.gradient(loss, self.actor_model.trainable_variables + self.critic_model.trainable_variables)
        self.actor_model.optimizer.apply_gradients(zip(grads[:len(self.actor_model.trainable_variables)], self.actor_model.trainable_variables))
        self.critic_model.optimizer.apply_gradients(zip(grads[len(self.actor_model.trainable_variables):], self.critic_model.trainable_variables))

    # Buffer'a veri ekleme
    def store_experience(self, state, action, reward, done, value, next_value):
        if len(self.buffer['states']) >= self.buffer_size:
            self.buffer['states'].pop(0)
            self.buffer['actions'].pop(0)
            self.buffer['rewards'].pop(0)
            self.buffer['dones'].pop(0)
            self.buffer['values'].pop(0)
            self.buffer['next_values'].pop(0)
        
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['dones'].append(done)
        self.buffer['values'].append(value)
        self.buffer['next_values'].append(next_value)


# Eğitim Fonksiyonu
def train_agent_ppo(agent, num_train_episodes):
    env = BlackjackEnv()
    total_wins = 0
    win_ratios = []

    start_time = time.time()  # Eğitim başlangıç zamanı
    for episode in range(num_train_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        states, actions, rewards, dones, values, next_values = [], [], [], [], [], []

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            value = agent.critic_model.predict(state.reshape(1, -1))[0][0]
            next_value = agent.critic_model.predict(next_state.reshape(1, -1))[0][0]
            
            # Buffer'a veri ekleme
            agent.store_experience(state, action, reward, done, value, next_value)

            total_reward += reward
            state = next_state

        # Burada buffer'dan veriler alınarak update işlemi yapılabilir
        advantages = agent.compute_advantages(agent.buffer['rewards'], agent.buffer['values'], agent.buffer['next_values'], agent.buffer['dones'])
        returns = np.array(agent.buffer['values']) + advantages
        agent.update(np.array(agent.buffer['states']), agent.buffer['actions'], advantages, returns)

        total_wins += (total_reward > 0)
        win_ratios.append(total_wins / (episode + 1))  # Toplam kazanma oranı

    end_time = time.time()  # Eğitim bitiş zamanı
    print(f"Eğitim süresi: {end_time - start_time:.2f} saniye")
    return win_ratios


# Test Fonksiyonu
def test_agent_ppo(agent, num_test_episodes):
    env = BlackjackEnv()
    total_wins = 0
    test_win_ratios = []

    start_time = time.time()  # Test başlangıç zamanı
    for episode in range(num_test_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state

        total_wins += (total_reward > 0)
        test_win_ratios.append(total_wins / (episode + 1))  # Toplam kazanma oranı

    end_time = time.time()  # Test bitiş zamanı
    print(f"Test süresi: {end_time - start_time:.2f} saniye")
    return test_win_ratios


# Kazanma Oranı Grafik Fonksiyonu (Eğitim ve Test ayrı ayrı sayfalarda)
def plot_training_win_ratio(train_win_ratios):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_win_ratios) + 1), train_win_ratios, label='PPO Train win rate', color='blue')
    plt.title("PPO Train win rate")
    plt.xlabel("Bölüm Sayısı")
    plt.ylabel("Kazanma Oranı")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)
    plt.show()

def plot_testing_win_ratio(test_win_ratios):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(test_win_ratios) + 1), test_win_ratios, label='PPO Test win rate', color='green')
    plt.title("PPO Test win rate")
    plt.xlabel("Bölüm Sayısı")
    plt.ylabel("Kazanma Oranı")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)
    plt.show()

# Ana Fonksiyon
def CreatePpo(num_train, num_test):
    agent = PPOAgent(action_size=2)
    train_win_ratios = train_agent_ppo(agent, num_train)
    test_win_ratios = test_agent_ppo(agent, num_test)
    plot_training_win_ratio(train_win_ratios)
    plot_testing_win_ratio(test_win_ratios)
###################################################Arayüz#################################################
print("*"*40+"Blackjack'e Hosgeldiniz"+"*"*40)
while True:
    num_train=0
    num_test=0
    print("Lutfen islem secin=>")
    print("[1]Sac egitim=>")
    print("[2]DDPQ egitim=>")
    print("[3]PPO egitim=>")
    print("[4]Quit=>")
    islem=""
    islem=(input())
    if(islem=="1"):   
        try:
            num_train=(int)(input("Lutfen egitim sayisini girin=>"))
            num_test=(int)(input("Lutfen test sayisini girin=>"))
            CreateSac(num_train,num_test)
        except ValueError:
            print("Hatali girdi!!!!!!")
    elif(islem=="2"):
        try:
            num_train=(int)(input("Lutfen egitim sayisini girin=>"))
            num_test=(int)(input("Lutfen test sayisini girin=>"))
            Createddpg(num_train,num_test)
        except ValueError:
            print("Hatali girdi!!!!!!")
    elif(islem=="3"):
        try:
            num_train=(int)(input("Lutfen egitim sayisini girin=>"))
            num_test=(int)(input("Lutfen test sayisini girin=>"))
            CreatePpo(num_train,num_test)
        except ValueError:
            print("Hatali girdi!!!!!!")
    elif(islem=="4"):
        break
    else: 
        print("Hatali islem seçildi!!!!")
print("*"*103)