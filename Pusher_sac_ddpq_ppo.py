import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
#python=3.8.0, tensorflow=2.8.0, numpy=1.21.0, gym=0.26.2, matplotlib=3.4.3
###############################################Pusher Environment##################################################
class PusherEnv:
    def __init__(self):
        # Başlangıç pozisyonları
        self.pusher_pos = np.array([5.0, 5.0])  # Ajan pozisyonu
        self.target_pos = np.array([7.0, 7.0])  # Hedef pozisyonu
        self.action_space = np.array([1.0, 1.0])  # Eylem alanı
        self.state = np.concatenate([self.pusher_pos, self.target_pos])  # State (Durum)

    def step(self, action):
        # Action'ı numpy array'e dönüştür ve şekil uyumu sağla
        action = np.array(action).flatten()
        self.pusher_pos += action  # Ajan hareket ediyor
        done = np.linalg.norm(self.pusher_pos - self.target_pos) < 0.1  # Ajan hedefe ulaştı mı
        reward = -np.linalg.norm(self.pusher_pos - self.target_pos)  # Ödül (yakınlık)
        next_state = np.concatenate([self.pusher_pos, self.target_pos])  # Yeni durum

        return next_state, reward, done, {}

    def reset(self):
        self.pusher_pos = np.array([5.0, 5.0])  # Ajan başlangıç pozisyonuna dönüyor
        self.state = np.concatenate([self.pusher_pos, self.target_pos])  # Yeni durum
        return self.state
    
    def render(self, initial=False, final=False):
        plt.figure(figsize=(6, 6))
        # Başlangıç konumları da çizilsin
        if initial:
            plt.plot(5.0, 5.0, 'bo', label='Pusher (Initial)', markersize=12)  # Başlangıç pusher
            plt.plot(7.0, 7.0, 'ro', label='Target (Initial)', markersize=12)  # Başlangıç target
        elif final:
            plt.plot(self.pusher_pos[0], self.pusher_pos[1], 'go', label='Pusher (Final)', markersize=12)  # Son Pusher
            plt.plot(self.target_pos[0], self.target_pos[1], 'ro', label='Target', markersize=12)  # Hedef
        else:
            plt.plot(self.pusher_pos[0], self.pusher_pos[1], 'bo', label='Pusher', markersize=12)  # Pusher (Ajan)
            plt.plot(self.target_pos[0], self.target_pos[1], 'ro', label='Target', markersize=12)  # Hedef
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend(loc="best")
        plt.title('Pusher and Target')
        plt.grid(True)
        plt.show()
#######################################################################################################
# SAC Ajanı
class SAC:
    def __init__(self, state_dim, action_dim):
        # Model ve optimizasyon ayarları
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor = self.build_actor()
        self.critic1 = self.build_critic()
        self.critic2 = self.build_critic()
        self.target_critic1 = self.build_critic()
        self.target_critic2 = self.build_critic()
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.gamma = 0.99  # Discount factor
        self.tau = 0.005   # Soft update for target networks

    def build_actor(self):
        model = models.Sequential([
            layers.Input(shape=(self.state_dim,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(self.action_dim, activation='tanh')  # Continuous action space
        ])
        return model

    def build_critic(self):
        model = models.Sequential([
            layers.Input(shape=(self.state_dim + self.action_dim,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(1)  # Value prediction
        ])
        return model

    def get_action(self, state):
        state = np.expand_dims(state, axis=0)
        action = self.actor(state)
        return action.numpy()[0]  # Action döndürülüyor

    def train(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, axis=0)
        next_state = np.expand_dims(next_state, axis=0)
        action = np.expand_dims(action, axis=0)

        with tf.GradientTape(persistent=True) as tape:
            # Critic1 ve Critic2 kaybı hesaplanıyor
            q1_value = self.critic1(tf.concat([state, action], axis=-1))
            q2_value = self.critic2(tf.concat([state, action], axis=-1))
            target_action = self.actor(next_state)
            target_q1 = self.target_critic1(tf.concat([next_state, target_action], axis=-1))
            target_q2 = self.target_critic2(tf.concat([next_state, target_action], axis=-1))
            target_q = reward + self.gamma * (1 - done) * tf.minimum(target_q1, target_q2)
            
            critic1_loss = tf.reduce_mean(tf.square(q1_value - target_q))
            critic2_loss = tf.reduce_mean(tf.square(q2_value - target_q))

            # Actor kaybı
            actor_loss = -tf.reduce_mean(self.critic1(tf.concat([state, self.actor(state)], axis=-1)))

        # Backpropagation for Critic1, Critic2, and Actor
        critic1_grads = tape.gradient(critic1_loss, self.critic1.trainable_variables)
        critic2_grads = tape.gradient(critic2_loss, self.critic2.trainable_variables)
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)

        self.critic_optimizer.apply_gradients(zip(critic1_grads, self.critic1.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic2_grads, self.critic2.trainable_variables))
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # Soft update for target networks
        self.update_target(self.target_critic1, self.critic1)
        self.update_target(self.target_critic2, self.critic2)

        return critic1_loss, critic2_loss, actor_loss

    def update_target(self, target, source):
        for target_var, source_var in zip(target.trainable_variables, source.trainable_variables):
            target_var.assign(target_var * (1 - self.tau) + source_var * self.tau)

def Create_Sac(train_num):
    # Ortamı ve ajanı başlatma
    env = PusherEnv()
    state_dim = len(env.state)  # state boyutunu almak için
    action_dim = len(env.action_space)  # action boyutunu almak için

    # SAC ajanını başlatıyoruz
    agent = SAC(state_dim, action_dim)

    # Eğitim süreci
    done = False
    state = env.state

    # Başlangıç durumunu çiziyoruz (ilk konumlar)
    env.render(initial=True)


    for step in range(train_num):
        # Ajanın vereceği aksiyonu al
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)  # Ortamda adım at

        # Ajanı eğit
        critic1_loss, critic2_loss, actor_loss = agent.train(state, action, reward, next_state, done)

        # Durum güncelleme
        state = next_state

        # Her 100 adımda bir terminale bilgi yazdır
        if step % 100 == 0:
            print(f"Step: {step}, Pusher Position: {env.pusher_pos}, Reward: {reward}")
        
        # Eğer ajan hedefe ulaştıysa sonlandır
        if done:
            print(f"Episode completed in {step+1} steps!")
            break

    # Eğitim sonu konumlarını çiz
    env.render(final=True)

################################################SAC Agent##############################################
# DDPQ Ajanı
class DDPG:
    def __init__(self, state_dim, action_dim, action_bound):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.target_actor = self.build_actor()
        self.target_critic = self.build_critic()
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.gamma = 0.99  # Discount factor
        self.tau = 0.005   # Soft update rate for target networks

    def build_actor(self):
        model = models.Sequential([
            layers.Input(shape=(self.state_dim,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(self.action_dim, activation='tanh'),
            layers.Lambda(lambda x: x * self.action_bound)  # Scale action output
        ])
        return model

    def build_critic(self):
        model = models.Sequential([
            layers.Input(shape=(self.state_dim + self.action_dim,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(1)  # Value prediction
        ])
        return model

    def get_action(self, state, noise=0.1):
        state = np.expand_dims(state, axis=0)
        action = self.actor(state).numpy()[0]
        action = np.clip(action + noise * np.random.randn(self.action_dim), -self.action_bound, self.action_bound)
        return action

    def train(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, axis=0)
        next_state = np.expand_dims(next_state, axis=0)
        action = np.expand_dims(action, axis=0)
        
        with tf.GradientTape(persistent=True) as tape:
            # Critic loss
            target_action = self.target_actor(next_state)
            target_q = reward + self.gamma * (1 - float(done)) * self.target_critic(tf.concat([next_state, target_action], axis=-1))
            q_value = self.critic(tf.concat([state, action], axis=-1))
            critic_loss = tf.reduce_mean(tf.square(q_value - target_q))

            # Actor loss
            actor_loss = -tf.reduce_mean(self.critic(tf.concat([state, self.actor(state)], axis=-1)))

        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # Soft update for target networks
        self.update_target(self.target_actor, self.actor)
        self.update_target(self.target_critic, self.critic)

        return critic_loss, actor_loss

    def update_target(self, target, source):
        for target_var, source_var in zip(target.trainable_variables, source.trainable_variables):
            target_var.assign(target_var * (1 - self.tau) + source_var * self.tau)

def Create_Ddpq(num_train):
    # Ortam ve DDPG ajanı başlatma
    env = PusherEnv()
    state_dim = len(env.state)
    action_dim = len(env.action_space)
    action_bound = 1.0

    agent = DDPG(state_dim, action_dim, action_bound)

    # Eğitim süreci
    done = False
    state = env.state

    env.render(initial=True)

    for step in range(num_train):
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        
        critic_loss, actor_loss = agent.train(state, action, reward, next_state, done)
        state = next_state

        if step % 100 == 0:
            print(f"Step: {step}, Pusher Position: {env.pusher_pos}, Reward: {reward}")

        if done:
            print(f"Episode completed in {step + 1} steps!")
            break

    env.render(final=True)

###################################################PPO Agent###########################################
# PPO Ajanı
class PPO:
    def __init__(self, state_dim, action_dim, action_bound):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
        self.gamma = 0.99  # Discount factor
        self.lamda = 0.95  # GAE lambda
        self.clip_ratio = 0.2  # PPO clip ratio
        self.max_grad_norm = 0.5  # Max gradient norm for clipping

    def build_actor(self):
        model = models.Sequential([
            layers.Input(shape=(self.state_dim,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(self.action_dim, activation='tanh'),
            layers.Lambda(lambda x: x * self.action_bound)  # Scale action output
        ])
        return model

    def build_critic(self):
        model = models.Sequential([
            layers.Input(shape=(self.state_dim,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(1)  # Value prediction
        ])
        return model

    def get_action(self, state):
        state = np.expand_dims(state, axis=0)  # shape: (1, state_dim)
        action = self.actor(state).numpy()[0]  # Take the first (and only) output
        return action

    def get_value(self, state):
        state = np.expand_dims(state, axis=0)  # shape: (1, state_dim)
        return self.critic(state).numpy()[0][0]

    def compute_advantages(self, rewards, values, dones):
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + (self.gamma * values[t + 1] if t < len(rewards) - 1 else 0) - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.lamda * (1 - dones[t]) * last_advantage
        return advantages

    def train(self, states, actions, rewards, next_states, dones, batch_size=64):
        values = np.array([self.get_value(s) for s in states])
        advantages = self.compute_advantages(rewards, values, dones)

        with tf.GradientTape() as tape:
            old_actions = np.array(actions)
            actions_pred = self.actor(states)
            ratio = tf.exp(tf.reduce_sum(actions_pred * old_actions, axis=1))  # Aksiyon oranı
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))

            value_pred = self.critic(states)
            critic_loss = tf.reduce_mean(tf.square(rewards - value_pred))

            total_loss = actor_loss + 0.5 * critic_loss

        grads = tape.gradient(total_loss, self.actor.trainable_variables + self.critic.trainable_variables)
        grads = [tf.clip_by_value(grad, -self.max_grad_norm, self.max_grad_norm) for grad in grads]
        self.actor_optimizer.apply_gradients(zip(grads[:len(self.actor.trainable_variables)], self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(grads[len(self.actor.trainable_variables):], self.critic.trainable_variables))

    def train_step(self, env, agent,num_train,max_steps=20):
        env.render(initial=True)  # Başlangıç konumunu göster

        for episode in range(num_train):
            state = env.reset()
            done = False
            rewards, states, actions, next_states, dones = [], [], [], [], []

            for step in range(max_steps):
                action = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)
                rewards.append(reward)
                states.append(state)
                actions.append(action)
                next_states.append(next_state)
                dones.append(done)
                state = next_state

                if done:
                    break

            states = np.array(states, dtype=np.float32)
            actions = np.array(actions, dtype=np.float32)
            next_states = np.array(next_states, dtype=np.float32)
            rewards = np.array(rewards, dtype=np.float32)
            dones = np.array(dones, dtype=np.float32)

            agent.train(states, actions, rewards, next_states, dones)

            # Her 100 bölümde bir konum ve ödülü yazdır
            if episode % 100 == 0:
                total_reward = sum(rewards)
                print(f"Episode: {episode}, Final Position: {env.pusher_pos}, Total Reward: {total_reward}")

        env.render(final=True)  # Eğitim sonrası son konumu göster
def Create_Ppo(num_train):
    # Çevre ve ajan başlat
    env = PusherEnv()
    agent = PPO(state_dim=4, action_dim=2, action_bound=1)
    agent.train_step(env, agent ,num_train)

###################################################Arayüz###############################################
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
        except ValueError:
            print("Hatali girdi!!!!!!")
        Create_Sac(num_train)
    elif(islem=="2"):
        try:
            num_train=(int)(input("Lutfen egitim sayisini girin=>"))
        except ValueError:
            print("Hatali girdi!!!!!!")
        Create_Ddpq(num_train)
    elif(islem=="3"):
        try:
            num_train=(int)(input("Lutfen egitim sayisini girin=>"))
        except ValueError:
            print("Hatali girdi!!!!!!")
        Create_Ppo(num_train)
    elif(islem=="4"):
        break
    else: 
        print("Hatali islem seçildi!!!!")
print("*"*103)