import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#python=3.8.0, tensorflow=2.8.0, numpy=1.21.0, gym=0.26.2, matplotlib=3.4.3
###############################################Blackjack Ortamı########################################

class BlackjackEnv(gym.Env):
    def __init__(self):
        super(BlackjackEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(2)  # 0: Hit, 1: Stick
        self.observation_space = gym.spaces.Box(low=0, high=31, shape=(2,), dtype=np.float32)
    
    def reset(self):
        self.player_sum = np.random.randint(12, 22)  # Player's initial hand
        self.dealer_card = np.random.randint(1, 11)   # Dealer's visible card
        return np.array([self.player_sum, self.dealer_card], dtype=np.float32)

    def step(self, action):
        if action == 0:  # Hit
            self.player_sum += np.random.randint(1, 11)
            if self.player_sum > 21:  # Bust
                return np.array([self.player_sum, self.dealer_card], dtype=np.float32), -1, True, {}
        else:  # Stick
            while self.dealer_card < 17:  # Dealer plays
                self.dealer_card += np.random.randint(1, 11)
            if self.dealer_card > 21 or self.player_sum > self.dealer_card:
                return np.array([self.player_sum, self.dealer_card], dtype=np.float32), 1, True, {}
            elif self.player_sum < self.dealer_card:
                return np.array([self.player_sum, self.dealer_card], dtype=np.float32), -1, True, {}
            else:
                return np.array([self.player_sum, self.dealer_card], dtype=np.float32), 0, True, {}
        return np.array([self.player_sum, self.dealer_card], dtype=np.float32), 0, False, {}
    
################################################SAC Agent##############################################
class SACAgent:
    def __init__(self, action_size):
        self.action_size = action_size
        self.gamma = 0.99
        self.alpha = 0.2
        self.tau = 0.005

        # Actor ve Critic için modeller oluşturma
        self.actor_model = self.build_actor()
        self.critic_model_1 = self.build_critic()
        self.critic_model_2 = self.build_critic()

    def build_actor(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(2,)),  # Blackjack'teki gözlemler
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    def build_critic(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(2,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    def act(self, state):
        state = state.reshape(1, -1)
        prob = self.actor_model.predict(state)[0]
        return np.random.choice(self.action_size, p=prob)

def train_agent_sac(num_train_episodes):
    env = BlackjackEnv()
    agent = SACAgent(action_size=env.action_space.n)
    train_win_rates = []
    total_wins = 0

    for episode in range(num_train_episodes):
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
        train_win_rates.append(win_rate)

    return train_win_rates

# Test fonksiyonunu da benzer şekilde güncelleyelim.
def test_agent_sac(agent, num_test_episodes):
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

        # Kazanılan oyunları say (pozitif ödüller)
        if total_reward > 0:
            total_wins += 1
        win_rate = total_wins / (episode + 1)
        test_win_rates.append(win_rate)

    return test_win_rates

# Eğitim ve test aşamaları için kazanma oranlarını ayrı grafiklerde görselleştirelim.
def plot_win_rates_sac(train_win_rates, test_win_rates):
    # Eğitim kazanma oranı grafiği
    plt.figure(figsize=(10, 5))
    plt.plot(train_win_rates, label='Train Win Rate', color='blue')
    plt.xlabel('Episodes')
    plt.ylabel('Win Rate')
    plt.title('SAC Agent Train Win Rate')
    plt.legend()
    plt.show()

    # Test kazanma oranı grafiği
    plt.figure(figsize=(10, 5))
    plt.plot(test_win_rates, label='Test Win Rate', color='green')
    plt.xlabel('Episodes')
    plt.ylabel('Win Rate')
    plt.title('SAC Agent Test Win Rate')
    plt.legend()
    plt.show()

def CreateSac(num_train,num_test):
    # SAC agenti oluştur ve eğit
    agent = SACAgent(action_size=2)
    train_win_rates = train_agent_sac(num_train)
    test_win_rates = test_agent_sac(agent, num_test)
    # Grafikle kazanma oranlarını göster
    plot_win_rates_sac(train_win_rates, test_win_rates)

###############################################DDPQ Agent#################################################
# DDPG Agent
class DDPGAgent:
    def __init__(self, action_size):
        self.action_size = action_size
        self.gamma = 0.99
        self.tau = 0.005

        # Actor ve Critic modelleri
        self.actor_model = self.build_actor()
        self.critic_model = self.build_critic()
        self.target_actor_model = self.build_actor()
        self.target_critic_model = self.build_critic()

        self.target_actor_model.set_weights(self.actor_model.get_weights())
        self.target_critic_model.set_weights(self.critic_model.get_weights())

    def build_actor(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(2,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    def build_critic(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(2,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    def update_target(self):
        new_weights = []
        for target, main in zip(self.target_actor_model.weights, self.actor_model.weights):
            new_weights.append(self.tau * main + (1 - self.tau) * target)
        self.target_actor_model.set_weights(new_weights)

        new_weights = []
        for target, main in zip(self.target_critic_model.weights, self.critic_model.weights):
            new_weights.append(self.tau * main + (1 - self.tau) * target)
        self.target_critic_model.set_weights(new_weights)

    def act(self, state):
        state = state.reshape(1, -1)
        action_probs = self.actor_model.predict(state)[0]
        return np.random.choice(self.action_size, p=action_probs)

# Eğitim fonksiyonu
def train_agent_ddpq(num_train_episodes):
    env = BlackjackEnv()
    agent = DDPGAgent(action_size=env.action_space.n)
    train_win_rates = []
    total_wins = 0

    for episode in range(num_train_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state

            # Ajanın ağırlıklarını güncelleme
            agent.update_target()

        # Kazanılan oyunları say (pozitif ödüller)
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

        # Kazanılan oyunları say (pozitif ödüller)
        if total_reward > 0:
            total_wins += 1
        win_rate = total_wins / (episode + 1)
        test_win_rates.append(win_rate)

    return test_win_rates
# Kazanma oranlarını grafikte göster
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

def Createddpq(num_train,num_test):
    # SAC agenti oluştur ve eğit
    agent = DDPGAgent(action_size=2)
    train_win_rates = train_agent_ddpq(num_train)
    # Test et ve kazanma oranlarını al
    test_win_rates = test_agent_ddpq(agent, num_test)
    # Eğitim ve test süreçlerini ayrı grafiklerde göster
    plot_win_rates_ddpq(train_win_rates, test_win_rates)

###################################################PPO Agent##############################################
# PPO Ajanı
class PPOAgent:
    def __init__(self, action_size, clip_ratio=0.2, gamma=0.99, learning_rate=0.001):
        self.action_size = action_size
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.learning_rate = learning_rate
        
        self.actor_model = self.build_actor()
        self.critic_model = self.build_critic()

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

# Eğitim Fonksiyonu
def train_agent__ppo(num_train_episodes):
    env = BlackjackEnv()
    agent = PPOAgent(action_size=env.action_space.n)

    win_ratios = []
    win_count = 0
    for episode in range(num_train_episodes):
        states, actions, rewards, dones, values, next_values = [], [], [], [], [], []
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            value = agent.critic_model.predict(state.reshape(1, -1))[0][0]
            next_value = agent.critic_model.predict(next_state.reshape(1, -1))[0][0]

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            values.append(value)
            next_values.append(next_value)

            total_reward += reward
            state = next_state

        advantages = agent.compute_advantages(rewards, values, next_values, dones)
        returns = np.array(values) + advantages
        agent.update(np.array(states), actions, advantages, returns)

        # Kazanma oranını her 100 bölümde bir hesapla
        win_count += (total_reward > 0)  # Eğer toplam ödül > 0 ise 1 (kazanç)
        if (episode + 1) % 100 == 0:
            win_ratio = win_count / 100  # Kazanma oranı
            win_ratios.append(win_ratio)
            win_count = 0  # Her 100 bölümde sıfırlanır

    return win_ratios

# Test Fonksiyonu
def test_agent_ppo(agent, num_test_episodes):
    env = BlackjackEnv()
    win_count = 0  # Kazanma sayısını takip etmek için
    test_win_ratios = []

    for episode in range(num_test_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state

        # Toplam ödül pozitifse kazandı kabul et
        win_count += (total_reward > 0)
        test_win_ratios.append(win_count / (episode + 1))  # Kazanma oranını bölüme göre hesapla

    return test_win_ratios

# Kazanma Oranı Grafik Fonksiyonu
def plot_training_win_ratio_ppo(train_win_ratios):
    plt.figure(figsize=(10, 6))

    # Eğitim kazanma oranını çiz
    plt.plot(range(100, len(train_win_ratios) * 100 + 1, 100), train_win_ratios, label='Eğitim Kazanma Oranı', color='blue')

    plt.title("Eğitim Kazanma Oranı Değişimi")
    plt.xlabel("Bölüm Sayısı")
    plt.ylabel("Kazanma Oranı")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)  # 0-1 arasında değerler gösterilsin
    plt.show()

def plot_testing_win_ratio_ppo(test_win_ratios):
    plt.figure(figsize=(10, 6))

    # Test kazanma oranını çiz
    plt.plot(range(1, len(test_win_ratios) + 1), test_win_ratios, label='Test Kazanma Oranı', color='red')

    plt.title("Test Kazanma Oranı Değişimi")
    plt.xlabel("Bölüm Sayısı")
    plt.ylabel("Kazanma Oranı")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)  # 0-1 arasında değerler gösterilsin
    plt.show()

def CreatePpo(num_train , num_test):
    agent = PPOAgent(action_size=2)
    train_win_ratios = train_agent__ppo(num_train)
    test_win_ratios = test_agent_ppo(agent, num_test)
    # Eğitim ve test kazanma oranlarını çiz
    plot_training_win_ratio_ppo(train_win_ratios)
    plot_testing_win_ratio_ppo(test_win_ratios)

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
            Createddpq(num_train,num_test)
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